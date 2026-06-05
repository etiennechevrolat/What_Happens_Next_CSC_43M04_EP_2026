#!/usr/bin/env python3
"""
Confusion matrix of the best from-scratch Track A model on the held-out
val_dir, saved to figures/fig_trackA_confusion.png.

By default it picks the checkpoint with the highest real_val_top1 in
eval_results.json (i.e. the best from-scratch model). Override with:
    python make_trackA_confusion.py best_model_v6_10.pt

Needs torch + processed_data/val2/val (same setup as eval_all_ckpts.py).
Run from the repo root.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from dataset.video_dataset import VideoFrameDataset, collect_video_samples  # noqa: E402
from train import build_model  # noqa: E402
from utils import build_transforms  # noqa: E402

VAL_DIR = REPO / "processed_data" / "val2" / "val"
FIG = REPO / "figures"
FIG.mkdir(exist_ok=True)


def pick_best_ckpt() -> str:
    evals = json.loads((REPO / "eval_results.json").read_text())
    evals = [e for e in evals if "real_val_top1" in e]
    best = max(evals, key=lambda e: e["real_val_top1"])
    print(f"best from-scratch ckpt: {best['path']} "
          f"({best['model_name']}, top1={best['real_val_top1']})")
    return best["path"]


def class_names(val_dir: Path) -> dict:
    names = {}
    for p in sorted(val_dir.iterdir()):
        if not p.is_dir():
            continue
        m = re.match(r"^(\d+)_(.*)$", p.name)
        if m:
            idx, name = int(m.group(1)), m.group(2)
        else:
            idx, name = len(names), p.name
        names[idx] = name.replace("_", " ")
    return names


def main():
    ckpt_name = sys.argv[1] if len(sys.argv) > 1 else pick_best_ckpt()
    ckpt = REPO / ckpt_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    raw = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(raw["config"])
    model = build_model(cfg)
    model.load_state_dict(raw["model_state_dict"])
    model.to(device).eval()

    pretrained_used = bool(raw.get("pretrained", cfg.model.get("pretrained", False)))
    tf = build_transforms(is_training=False, use_imagenet_norm=pretrained_used)
    num_frames = int(raw.get("num_frames", cfg.dataset.num_frames))

    samples = collect_video_samples(VAL_DIR)
    ds = VideoFrameDataset(VAL_DIR, num_frames=num_frames, transform=tf,
                           sample_list=samples)
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=4,
                        pin_memory=(device.type == "cuda"))

    C = int(cfg.model.num_classes)
    cm = np.zeros((C, C), dtype=np.int64)
    with torch.no_grad():
        for vb, lbl in loader:
            pred = model(vb.to(device)).argmax(1).cpu().numpy()
            for t, p in zip(lbl.numpy(), pred):
                cm[t, p] += 1

    acc = np.trace(cm) / max(cm.sum(), 1)
    cmn = cm / np.clip(cm.sum(1, keepdims=True), 1, None)  # row-normalise

    # top-6 off-diagonal (directed) confused pairs
    off = cmn.copy()
    np.fill_diagonal(off, 0.0)
    flat = np.argsort(off.ravel())[::-1][:6]
    pairs = [(int(i // C), int(i % C), float(off[i // C, i % C])) for i in flat]

    names = class_names(VAL_DIR)
    short = [names.get(i, str(i))[:16] for i in range(C)]

    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    im = ax.imshow(cmn, cmap="viridis", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label="row-normalised frequency")
    ax.set_xticks(range(C)); ax.set_yticks(range(C))
    ax.set_xticklabels(short, rotation=90, fontsize=5)
    ax.set_yticklabels(short, fontsize=5)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"Track A best from-scratch confusion "
                 f"({raw.get('model_name', '?')}, top-1={acc:.3f})", fontsize=10)
    for (t, p, v) in pairs:
        ax.plot(p, t, "x", color="red", markersize=8, markeredgewidth=2)
    fig.tight_layout()
    out = FIG / "fig_trackA_confusion.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"wrote {out}  (top-1={acc:.4f})")
    print("most-confused pairs (true -> pred, freq):")
    for (t, p, v) in pairs:
        print(f"  {t:2d} {names.get(t, '')!r:24} -> {p:2d} {names.get(p, '')!r:24} {v:.2f}")


if __name__ == "__main__":
    main()
