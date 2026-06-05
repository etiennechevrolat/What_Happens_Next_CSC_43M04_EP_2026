"""
Evalue tous les best_model*.pt sur le REAL val (processed_data/val2/val) et
écrit un JSON avec top-1, top-5, params, latency.

Lancer depuis la racine du repo:
    cd /users/eleves-a/2024/etienne.chevrolat/What_Happens_Next_CSC_43M04_EP_2026
    source .venv/bin/activate
    PYTHONPATH=src python eval_all_ckpts.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

from dataset.video_dataset import VideoFrameDataset, collect_video_samples  # noqa: E402
from train import build_model  # noqa: E402
from utils import build_transforms  # noqa: E402


VAL_DIR = REPO / "processed_data" / "val2" / "val"
OUT_JSON = REPO / "eval_results.json"


def evaluate_ckpt(ckpt_path: Path, device: torch.device, batch_size: int = 16) -> dict:
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "config" not in raw or raw["config"] is None:
        return {"path": str(ckpt_path), "error": "no config in checkpoint"}

    cfg = OmegaConf.create(raw["config"])
    model = build_model(cfg)
    model.load_state_dict(raw["model_state_dict"])
    model.to(device).eval()

    n_params = sum(p.numel() for p in model.parameters())

    pretrained_used = bool(raw.get("pretrained", cfg.model.get("pretrained", False)))
    eval_transform = build_transforms(is_training=False, use_imagenet_norm=pretrained_used)
    num_frames = int(raw.get("num_frames", cfg.dataset.num_frames))

    samples = collect_video_samples(VAL_DIR)
    ds = VideoFrameDataset(
        root_dir=VAL_DIR,
        num_frames=num_frames,
        transform=eval_transform,
        sample_list=samples,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=(device.type == "cuda"))

    # Latence : 1 warmup batch + chrono moyen sur 5 batchs
    n_warm = 1
    latencies = []
    correct1 = 0
    correct5 = 0
    total = 0

    with torch.no_grad():
        for i, (vb, lbl) in enumerate(loader):
            vb = vb.to(device)
            lbl = lbl.to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            logits = model(vb)
            if device.type == "cuda":
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            if i >= n_warm and len(latencies) < 5:
                latencies.append(dt / vb.size(0))

            pred1 = logits.argmax(dim=1)
            correct1 += int((pred1 == lbl).sum().item())
            _, top5 = logits.topk(min(5, logits.size(1)), dim=1)
            correct5 += int(top5.eq(lbl.view(-1, 1)).any(dim=1).sum().item())
            total += lbl.size(0)

    top1 = correct1 / max(total, 1)
    top5 = correct5 / max(total, 1)
    lat_ms = (sum(latencies) / max(len(latencies), 1)) * 1000.0

    # Méta utile
    saved_acc = raw.get("real_val_accuracy", raw.get("val_accuracy"))
    model_name = raw.get("model_name", cfg.model.get("name"))

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "path": ckpt_path.name,
        "model_name": str(model_name),
        "num_frames": num_frames,
        "params_M": round(n_params / 1e6, 2),
        "real_val_top1": round(top1, 4),
        "real_val_top5": round(top5, 4),
        "saved_acc_in_ckpt": saved_acc,
        "fwd_latency_ms_per_clip": round(lat_ms, 2),
        "n_val_samples": total,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)

    ckpts = sorted(REPO.glob("best_model*.pt")) + sorted(REPO.glob("model_v6_*control*.pt"))
    print(f"Found {len(ckpts)} checkpoints", flush=True)

    results = []
    for ck in ckpts:
        print(f"--- {ck.name} ---", flush=True)
        try:
            r = evaluate_ckpt(ck, device)
            print(json.dumps(r, indent=2), flush=True)
            results.append(r)
        except Exception as e:
            print(f"FAILED {ck.name}: {e}", flush=True)
            results.append({"path": ck.name, "error": str(e)})

        # write incrementally
        OUT_JSON.write_text(json.dumps(results, indent=2))

    print(f"\nWrote {OUT_JSON}")


if __name__ == "__main__":
    main()
