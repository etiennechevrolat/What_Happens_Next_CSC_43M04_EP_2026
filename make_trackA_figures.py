#!/usr/bin/env python3
"""
Generate Track A (from-scratch) figures for report.tex.

Pure JSON -> matplotlib (no torch / no GPU / no dataset needed), so it runs
anywhere. Reads:
    eval_results.json    (held-out val top-1/top-5, params, latency per ckpt)
    wandb_curves.json    (MAE pretrain loss + model7_A finetune curves)

Outputs (300 dpi PNG) into figures/:
    fig_trackA_acc_vs_params.png   accuracy vs #params, from-scratch families
    fig_trackA_model7_curves.png   model7_A finetune: LLDR effect + gen. gap
    fig_trackA_mae_pretrain.png    self-supervised MAE reconstruction loss

Run from the repo root:
    python make_trackA_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parent
FIG = REPO / "figures"
FIG.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
})

# Display names for the from-scratch Track A families.
FAMILY = {
    "model1.2_A": r"CNN$\to$ViT$_t$ (m1.2)",
    "model2_A":   r"CNN$\to$ST-ViT (m2)",
    "model5_B":   r"R(2+1)D+ST-ViT (m5B)",
    "model7_A":   r"MAE ViT-B (m7)",
}
ORDER = ["model1.2_A", "model2_A", "model5_B", "model7_A"]
COLORS = {
    "model1.2_A": "#4C72B0",
    "model2_A":   "#55A868",
    "model5_B":   "#C44E52",
    "model7_A":   "#8172B3",
}

# Pretrained reference points (from Track B), for context only.
BASELINES = {
    "VideoMAE-B (SSV2 pretrain)": 0.6096,
    "V-JEPA-2 ViT-L (best Track B)": 0.638,
}


def load_json(name):
    return json.loads((REPO / name).read_text())


# ───────────────────────── Figure 1: acc vs params ──────────────────────────
def fig_acc_vs_params(evals):
    by_fam: dict[str, list[dict]] = {}
    for r in evals:
        if "real_val_top1" not in r:
            continue
        by_fam.setdefault(r["model_name"], []).append(r)

    fig, ax = plt.subplots(figsize=(4.0, 3.4))

    # baselines as horizontal reference lines
    for i, (lbl, acc) in enumerate(BASELINES.items()):
        ax.axhline(acc, ls="--", lw=1.0, color="0.55", zorder=0)
        ax.text(0.98, acc + 0.004, lbl, transform=ax.get_yaxis_transform(),
                ha="right", va="bottom", fontsize=6.5, color="0.4")

    for fam in ORDER:
        runs = by_fam.get(fam, [])
        if not runs:
            continue
        xs = [r["params_M"] for r in runs]
        ys = [r["real_val_top1"] for r in runs]
        c = COLORS[fam]
        # individual runs (small, transparent)
        ax.scatter(xs, ys, s=14, color=c, alpha=0.35, zorder=2)
        # family best (large marker)
        bi = int(np.argmax(ys))
        ax.scatter([xs[bi]], [ys[bi]], s=85, color=c, edgecolor="k",
                   linewidth=0.6, zorder=3, label=f"{FAMILY[fam]}  ({ys[bi]:.3f})")

    ax.set_xscale("log")
    ax.set_xlabel("trainable parameters (M, log scale)")
    ax.set_ylabel("held-out top-1 (val_dir)")
    ax.set_ylim(0.18, 0.68)
    ax.set_title("Track A: from-scratch accuracy vs. capacity", fontsize=9)
    ax.legend(loc="center right", fontsize=6.8, framealpha=0.9)
    out = FIG / "fig_trackA_acc_vs_params.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


# ─────────────────── Figure 2: model7_A finetune curves ─────────────────────
def fig_model7_curves(curves):
    ft = curves["finetune"]

    def cols(run):
        a = np.array(ft[run], dtype=float)
        # [epoch, train_top1(mixup), internal_val_top1, real_val_top1, lr]
        return a[:, 0], a[:, 1], a[:, 2], a[:, 3]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.2, 3.2))

    # --- Left: effect of LLDR on held-out accuracy ---
    regimes = [
        ("ft_mae70_NO_lldr", "no LLDR", "#999999"),
        ("ft_mae70_lldr_v2", "LLDR + warmup", "#4C72B0"),
        ("ft_diverse_resume", "LLDR + diverse aug (best)", "#C44E52"),
    ]
    for run, lbl, c in regimes:
        if run not in ft:
            continue
        e, _, _, rv = cols(run)
        axL.plot(e, rv, color=c, lw=1.8, label=f"{lbl}  ($\\rightarrow${rv[-1]:.3f})")
    axL.set_xlabel("epoch")
    axL.set_ylabel("held-out top-1 (val_dir)")
    axL.set_title("LLDR is decisive for the MAE ViT-B", fontsize=9)
    axL.legend(loc="lower right", fontsize=6.8)

    # --- Right: generalization gap on a clean full run ---
    run = "ft_mae70_lldr_v2"
    e, tr, iv, rv = cols(run)
    axR.plot(e, tr, color="#DD8452", lw=1.6, label="train top-1 (MixUp inputs)")
    axR.plot(e, iv, color="#55A868", lw=1.6, label="internal-val top-1")
    axR.plot(e, rv, color="#C44E52", lw=1.6, label="held-out top-1 (val_dir)")
    axR.fill_between(e, rv, iv, color="#55A868", alpha=0.12)
    gap = iv[-1] - rv[-1]
    axR.annotate(f"gap $\\approx${gap*100:.0f} pts",
                 xy=(e[-1], (iv[-1] + rv[-1]) / 2),
                 xytext=(e[-1] - 18, (iv[-1] + rv[-1]) / 2 + 0.06),
                 fontsize=7.5, color="0.25",
                 arrowprops=dict(arrowstyle="->", color="0.4", lw=0.8))
    axR.set_xlabel("epoch")
    axR.set_ylabel("top-1")
    axR.set_title("internal-val vs. held-out gap", fontsize=9)
    axR.legend(loc="lower right", fontsize=6.8)

    fig.suptitle("model7_A — VideoMAE-v2 pretrained ViT-B finetune (4 frames)",
                 fontsize=9.5)
    out = FIG / "fig_trackA_model7_curves.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


# ─────────────────── Figure 3: MAE pretraining loss ─────────────────────────
def fig_mae_pretrain(curves):
    pt = curves["pretrain"]

    def series(run):
        a = np.array(pt[run], dtype=float)
        return a[:, 0], a[:, 1]

    fig, ax = plt.subplots(figsize=(4.4, 3.2))

    # stable runs
    plotted = []
    for run, lbl, c in [
        ("mae_fresh_mask075", "mask 0.75 (fresh)", "#4C72B0"),
        ("mae_mask075_orig", "mask 0.75 (long)", "#55A868"),
        ("mae_mask09_v2_first", "mask 0.90 (stabilized)", "#DD8452"),
    ]:
        if run not in pt:
            continue
        e, l = series(run)
        ax.plot(e, l, color=c, lw=1.6, label=lbl)
        plotted.append(run)

    # continue the stabilized 0.90 run (resumed segments), same colour
    for run in ("mae_mask09_v2_part2", "mae_mask09_v2_part3"):
        if run in pt:
            e, l = series(run)
            ax.plot(e, l, color="#DD8452", lw=1.6)

    # divergent high-mask run -> NaN
    if "mae_mask09_orig_nan" in pt:
        e, l = series("mae_mask09_orig_nan")
        ok = ~np.isnan(l)
        ax.plot(e[ok], l[ok], color="#C44E52", lw=1.6, ls=":",
                label="mask 0.90 (diverged)")
        # mark first NaN
        nan_idx = np.where(np.isnan(l))[0]
        if len(nan_idx):
            xd = e[nan_idx[0]]
            ax.axvline(xd, color="#C44E52", lw=0.8, ls=":")
            ax.scatter([e[ok][-1]], [l[ok][-1]], marker="x", color="#C44E52", s=45,
                       zorder=5)
            ax.text(xd + 1, l[ok][-1], "NaN", color="#C44E52", fontsize=7,
                    va="center")

    ax.set_xlabel("pretrain epoch")
    ax.set_ylabel("norm-pixel MSE reconstruction loss")
    ax.set_title("Self-supervised VideoMAE-v2 pretraining (from scratch)", fontsize=9)
    ax.legend(loc="upper right", fontsize=6.8)
    out = FIG / "fig_trackA_mae_pretrain.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    evals = load_json("eval_results.json")
    curves = load_json("wandb_curves.json")
    fig_acc_vs_params(evals)
    fig_model7_curves(curves)
    fig_mae_pretrain(curves)
    print("done.")


if __name__ == "__main__":
    main()
