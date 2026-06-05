#!/usr/bin/env python3
"""
Run a trained checkpoint on the test split and write a submission CSV::

    video_name,predicted_class

Uses the same Hydra layout as ``train.py`` / ``evaluate.py``. Paths and checkpoint
come from the composed config (see ``configs/data/default.yaml`` and
``configs/train/default.yaml``).

Example (from ``src/``)::

    python create_submission.py
    python create_submission.py training.checkpoint_path=/path/to/best_model.pt
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
import torch.nn.functional as F_torch

from dataset.video_dataset import VideoFrameDataset
from train import build_model
from utils import build_transforms, set_seed


def load_manifest_video_names(manifest_path: Path) -> List[str]:
    with manifest_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "video_name" not in reader.fieldnames:
            raise ValueError(f"{manifest_path} must contain a 'video_name' column.")
        return [row["video_name"].strip() for row in reader]


def _index_video_folders(test_root: Path) -> Dict[str, Path]:
    """
    Walk ``test_root`` **once** and map each ``video_<id>`` folder name -> path.

    Prunes search at each ``video_*`` directory (frames live there; no need to
    descend), so we avoid scanning every JPEG.

    This replaces ``glob(f'**/{name}')`` per manifest row, which re-walked the
    full tree thousands of times.
    """
    test_root = test_root.resolve()
    index: Dict[str, Path] = {}
    for dirpath, dirs, _files in os.walk(test_root, topdown=True):
        base = Path(dirpath)
        for name in list(dirs):
            if not name.startswith("video_"):
                continue
            p = (base / name).resolve()
            if name in index:
                raise FileNotFoundError(
                    f"Duplicate video folder name {name!r}: {index[name]} and {p}"
                )
            index[name] = p
            dirs.remove(name)
    return index


def resolve_video_dirs(test_root: Path, video_names: List[str]) -> List[Path]:
    """Map each ``video_<id>`` folder name to a path using a pre-built index."""
    index = _index_video_folders(test_root)
    out: List[Path] = []
    missing: List[str] = []
    for name in video_names:
        p = index.get(name)
        if p is None:
            missing.append(name)
        else:
            out.append(p)
    if missing:
        sample = ", ".join(repr(m) for m in missing[:5])
        extra = f" (+{len(missing) - 5} more)" if len(missing) > 5 else ""
        raise FileNotFoundError(
            f"{len(missing)} manifest video(s) not found under {test_root}: {sample}{extra}"
        )
    return out


def discover_all_test_videos(test_root: Path) -> Tuple[List[str], List[Path]]:
    """
    Discover all ``video_*`` folders under ``test_root`` and return them sorted.

    Returns:
        (video_names, video_dirs) sorted by video folder name.
    """
    index = _index_video_folders(test_root)
    video_names = sorted(index.keys())
    video_dirs = [index[name] for name in video_names]
    return video_names, video_dirs


def build_model_from_checkpoint(ckpt: Dict[str, Any]) -> torch.nn.Module:
    """Rebuild the model using the saved Hydra config when available."""
    if "config" in ckpt and ckpt["config"] is not None:
        cfg = OmegaConf.create(ckpt["config"])
        return build_model(cfg)

    cfg = OmegaConf.create(
        {
            "model": {
                "name": ckpt.get("model_name", "cnn_baseline"),
                "num_classes": int(ckpt["num_classes"]),
                "pretrained": bool(ckpt.get("pretrained", True)),
                "lstm_hidden_size": int(ckpt.get("lstm_hidden_size", 512)),
            }
        }
    )
    return build_model(cfg)


def _spatial_crops(video: torch.Tensor, num_crops: int, crop_frac: float = 0.857) -> List[torch.Tensor]:
    """
    Multi-crop spatial TTA. video: (B, T, C, H, W) avec H=W=out_size.
    Découpe `num_crops` patchs (192x192 si out_size=224 et crop_frac=0.857)
    aux positions {center}, {TL,center,BR} ou {TL,TR,BL,BR,center}, puis resize
    chacun à la taille d'origine pour passer dans le modèle.
    Retourne une liste de tenseurs de même shape que `video`.
    """
    B, T, C, H, W = video.shape
    out_size = H
    cs = int(round(H * crop_frac))
    cs = max(1, min(cs, H))
    if cs == H or num_crops <= 1:
        return [video]

    di = H - cs
    dj = W - cs
    mid_i, mid_j = di // 2, dj // 2
    if num_crops == 3:
        offsets = [(0, 0), (mid_i, mid_j), (di, dj)]
    elif num_crops >= 5:
        offsets = [(0, 0), (0, dj), (di, 0), (di, dj), (mid_i, mid_j)]
    else:
        offsets = [(mid_i, mid_j)]

    flat = video.reshape(B * T, C, H, W)
    crops = []
    for (i, j) in offsets:
        c = flat[..., i:i + cs, j:j + cs]
        c = F_torch.interpolate(c, size=(out_size, out_size), mode="bilinear", align_corners=False)
        crops.append(c.reshape(B, T, C, out_size, out_size))
    return crops


@torch.no_grad()
def run_inference(
    models: List[torch.nn.Module],
    loader: DataLoader,
    device: torch.device,
    total_videos: int,
    use_tta_flip: bool = True,
    num_crops: int = 5,
    crop_frac: float = 0.857,
) -> List[int]:
    """
    Ensemble : chaque modèle prédit, et chaque prédiction passe par TTA flip
    + multi-crop spatial. Renvoie les argmax sur la moyenne des softmax
    (n_models × n_crops × {1,2} passes au total).
    """
    for m in models:
        m.eval()

    preds = []
    n_batches = len(loader)
    log_interval = max(1, n_batches // 10)
    processed = 0

    for batch_idx, (video_batch, _labels) in enumerate(loader, start=1):
        video_batch = video_batch.to(device)
        crops = _spatial_crops(video_batch, num_crops=num_crops, crop_frac=crop_frac)

        probs_sum = None
        n_passes = 0
        for crop in crops:
            crop_flip = torch.flip(crop, dims=[-1]) if use_tta_flip else None
            for m in models:
                p = F_torch.softmax(m(crop), dim=1)
                probs_sum = p if probs_sum is None else probs_sum + p
                n_passes += 1
                if use_tta_flip:
                    probs_sum = probs_sum + F_torch.softmax(m(crop_flip), dim=1)
                    n_passes += 1

        probs = probs_sum / n_passes
        preds.extend(int(p) for p in probs.argmax(dim=1).cpu().tolist())

        processed += video_batch.size(0)
        if batch_idx % log_interval == 0 or batch_idx == n_batches:
            print(f"  Inference batch {batch_idx}/{n_batches} ({processed}/{total_videos})", flush=True)

    return preds


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    set_seed(int(cfg.dataset.seed))

    device_str = cfg.training.device
    if device_str == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; using CPU.")
        device_str = "cpu"
    device = torch.device(device_str)


    # Liste de checkpoints (Hydra : peut être string unique ou liste)
    ckpt_cfg = cfg.training.get("checkpoints", None)
    if ckpt_cfg is None:
        ckpt_paths = [Path(cfg.training.checkpoint_path).resolve()]
    else:
        ckpt_paths = [Path(str(p)).resolve() for p in ckpt_cfg]

    for p in ckpt_paths:
        if not p.is_file():
            raise SystemExit(f"Checkpoint not found: {p}")

    print(f"Loading {len(ckpt_paths)} checkpoint(s):", flush=True)
    models = []
    num_frames_ref, pretrained_ref, num_classes_ref = None, None, None
    for p in ckpt_paths:
        print(f"  - {p}", flush=True)
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        m = build_model_from_checkpoint(ckpt)
        m.load_state_dict(ckpt["model_state_dict"])
        m.to(device)
        models.append(m)
        nf = int(ckpt.get("num_frames", cfg.dataset.num_frames))
        pre = bool(ckpt.get("pretrained", cfg.model.pretrained))
        nc = int(ckpt.get("num_classes", cfg.model.num_classes))
        if num_frames_ref is None:
            num_frames_ref, pretrained_ref, num_classes_ref = nf, pre, nc
        elif (nf, pre, nc) != (num_frames_ref, pretrained_ref, num_classes_ref):
            raise SystemExit(f"Incompatible checkpoint {p}: ({nf},{pre},{nc}) vs ({num_frames_ref},{pretrained_ref},{num_classes_ref})")

    num_frames = num_frames_ref
    pretrained = pretrained_ref
    eval_transform = build_transforms(is_training=False, use_imagenet_norm=pretrained)

    test_root = Path(cfg.dataset.test_dir).resolve()
    output_path = Path(cfg.dataset.submission_output).resolve()
    manifest_cfg = cfg.dataset.get("test_manifest")

    print(f"Indexing video folders under: {test_root}", flush=True)
    if manifest_cfg:
        manifest_path = Path(str(manifest_cfg)).resolve()
        print(f"Reading manifest: {manifest_path}", flush=True)
        video_names = load_manifest_video_names(manifest_path)
        video_dirs = resolve_video_dirs(test_root, video_names)
        print(
            f"Resolved {len(video_dirs)} video folders from manifest for inference.",
            flush=True,
        )
    else:
        print(
            "No dataset.test_manifest provided; using all video_* folders found in test_dir.",
            flush=True,
        )
        video_names, video_dirs = discover_all_test_videos(test_root)
        print(
            f"Discovered {len(video_dirs)} video folders (sorted by video name).",
            flush=True,
        )
    sample_list: List[Tuple[Path, int]] = [(p, 0) for p in video_dirs]

    dataset = VideoFrameDataset(
        root_dir=test_root,
        num_frames=num_frames,
        transform=eval_transform,
        sample_list=sample_list,
    )
    batch_size = int(cfg.training.batch_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(cfg.training.num_workers),
        pin_memory=(device.type == "cuda"),
    )

    print(
        f"Starting inference: {len(dataset)} clips, batch_size={batch_size}, "
        f"{len(loader)} batches",
        flush=True,
    )
    num_crops = int(cfg.training.get("num_crops", 5))
    crop_frac = float(cfg.training.get("crop_frac", 0.857))
    use_tta_flip = bool(cfg.training.get("use_tta_flip", True))
    print(f"TTA: num_crops={num_crops}, crop_frac={crop_frac}, hflip={use_tta_flip}", flush=True)
    predictions = run_inference(
        models, loader, device, total_videos=len(dataset),
        use_tta_flip=use_tta_flip, num_crops=num_crops, crop_frac=crop_frac,
    )
    print("Inference finished.", flush=True)

    if len(predictions) != len(video_names):
        raise RuntimeError(
            f"Prediction count {len(predictions)} != manifest length {len(video_names)}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing submission CSV: {output_path}", flush=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video_name", "predicted_class"])
        for name, pred in zip(video_names, predictions):
            w.writerow([name, pred])

    print(f"Done. Wrote {len(predictions)} rows to {output_path}", flush=True)


if __name__ == "__main__":
    main()
