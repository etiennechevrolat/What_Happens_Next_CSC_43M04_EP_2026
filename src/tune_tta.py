"""
Tune TTA hyperparameters (num_crops, crop_frac, use_flip) on the val set.

For each crop_frac, runs ONE pass over val computing softmax for each crop
and each flip independently, stores them, then offline tests all subsets
(different num_crops and flip on/off) to report val accuracy.

Usage:
    python src/tune_tta.py training.checkpoint_path=best_model_v7_lldr_diverse_snap.pt experiment=model7_A_experiment
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import hydra
import torch
import torch.nn.functional as F_torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from dataset.video_dataset import VideoFrameDataset, collect_video_samples
from train import build_model
from utils import build_transforms, set_seed
from create_submission import _spatial_crops


def load_model(ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = OmegaConf.create(ck["config"])
    model = build_model(cfg)
    model.load_state_dict(ck["model_state_dict"])
    model.to(device).eval()
    return model, cfg


@torch.no_grad()
def collect_logits(model, loader, device, num_crops: int, crop_frac: float, max_batches: int = None):
    """
    Returns: per_crop_flip_softmax (n_samples, n_crops, 2, n_classes), labels (n_samples,)
    Index [:, :, 0, :] = no flip, [:, :, 1, :] = flip
    """
    all_softmax = []
    all_labels = []
    for bi, (video_batch, labels) in enumerate(loader, start=1):
        video_batch = video_batch.to(device)
        labels = labels.to(device)
        crops = _spatial_crops(video_batch, num_crops=num_crops, crop_frac=crop_frac)
        # crops: list of length n_crops, each (B, T, C, H, W)
        B = video_batch.size(0)
        nC = len(crops)
        per_sample = torch.zeros(B, nC, 2, model.head.out_features if hasattr(model, 'head') else 33, device=device)
        # Get number of classes from a forward pass
        sample_out = model(crops[0])
        n_classes = sample_out.size(1)
        per_sample = torch.zeros(B, nC, 2, n_classes, device=device)
        per_sample[:, 0, 0] = F_torch.softmax(sample_out, dim=1)
        # First crop, no flip already done above
        # First crop, flip
        per_sample[:, 0, 1] = F_torch.softmax(model(torch.flip(crops[0], dims=[-1])), dim=1)
        # Remaining crops
        for ci in range(1, nC):
            per_sample[:, ci, 0] = F_torch.softmax(model(crops[ci]), dim=1)
            per_sample[:, ci, 1] = F_torch.softmax(model(torch.flip(crops[ci], dims=[-1])), dim=1)
        all_softmax.append(per_sample.cpu())
        all_labels.append(labels.cpu())
        if bi % 20 == 0:
            print(f"  batch {bi}", flush=True)
        if max_batches is not None and bi >= max_batches:
            break
    return torch.cat(all_softmax, 0), torch.cat(all_labels, 0)


def eval_subset(softmax_tensor, labels, num_crops_subset: int, use_flip: bool) -> float:
    """
    softmax_tensor: (N, total_crops, 2, C)
    Average over first num_crops_subset crops and over flip dim if use_flip.
    """
    sub = softmax_tensor[:, :num_crops_subset]  # (N, nc, 2, C)
    if use_flip:
        avg = sub.mean(dim=(1, 2))
    else:
        avg = sub[:, :, 0, :].mean(dim=1)
    preds = avg.argmax(dim=1)
    acc = (preds == labels).float().mean().item()
    return acc


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(int(cfg.dataset.seed))

    ckpt_path = Path(cfg.training.checkpoint_path).resolve()
    print(f"Loading: {ckpt_path}")
    model, ck_cfg = load_model(ckpt_path, device)

    # Build val dataset (with labels, unlike submission)
    val_dir = Path(cfg.dataset.val_dir).resolve()
    samples = collect_video_samples(val_dir)
    print(f"Val samples: {len(samples)}")

    # Use eval transform (no train augs)
    num_frames = int(ck_cfg.get("dataset", {}).get("num_frames", cfg.dataset.num_frames))
    pretrained = bool(ck_cfg.get("model", {}).get("pretrained", False))
    eval_tf = build_transforms(is_training=False, use_imagenet_norm=pretrained)
    ds = VideoFrameDataset(
        root_dir=str(val_dir),
        num_frames=num_frames,
        transform=eval_tf,
        sample_list=samples,
    )
    loader = DataLoader(ds, batch_size=int(cfg.training.get("batch_size", 24)),
                        shuffle=False, num_workers=4, pin_memory=True)

    CROP_FRACS = [1.0, 0.9, 0.857]
    MAX_NUM_CROPS = 5  # 4 corners + center
    print(f"\nWill test crop_fracs={CROP_FRACS} × num_crops in [1,3,5] × flip in [F,T]")

    results = []
    for cf in CROP_FRACS:
        print(f"\n=== crop_frac={cf} ===")
        try:
            sft, lbls = collect_logits(model, loader, device, num_crops=MAX_NUM_CROPS, crop_frac=cf)
        except Exception as e:
            print(f"  skipped: {e}")
            continue
        for nc in [1, 3, 5]:
            for flip in [False, True]:
                acc = eval_subset(sft, lbls, nc, flip)
                results.append((cf, nc, flip, acc))
                print(f"  cf={cf:.3f} nc={nc} flip={flip}: val_acc={acc:.4f}")

    print("\n=== TOP 10 ===")
    for r in sorted(results, key=lambda x: -x[3])[:10]:
        print(f"  cf={r[0]:.3f} nc={r[1]} flip={r[2]}: val_acc={r[3]:.4f}")


if __name__ == "__main__":
    main()
