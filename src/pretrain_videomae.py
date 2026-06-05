"""
Pretrain self-supervisé VideoMAE V2.

Usage (single GPU) :
    cd src && python pretrain_videomae.py experiment=videomae_v2_pretrain

Multi-GPU (torchrun) :
    cd src && torchrun --nproc_per_node=4 pretrain_videomae.py \
        experiment=videomae_v2_pretrain training.batch_size=32

À la fin du pretrain, on sauvegarde le state_dict complet de VideoMAEv2.
Pour transférer dans model6 :

    >>> from models.videomae_v2 import VideoMAEv2
    >>> from models.model6_A import VideoMaxViT
    >>> mae = VideoMAEv2(...same args...)
    >>> mae.load_state_dict(torch.load("videomae_v2_pretrain.pt")["model_state_dict"])
    >>> m6 = VideoMaxViT(...)
    >>> mae.transfer_to_model6_stage3(m6, num_blocks_to_transfer=4)
    >>> # puis finetune m6 avec train.py experiment=model6_A_experiment
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import hydra
import torch
import torch.distributed as dist
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.amp import autocast, GradScaler

from dataset.video_dataset import VideoFrameDataset, collect_video_samples
from models.videomae_v2 import VideoMAEv2
from utils import build_transforms, set_seed


def get_module(m):
    return m.module if isinstance(m, DDP) else m


def build_mae(cfg: DictConfig, num_frames: int) -> VideoMAEv2:
    return VideoMAEv2(
        img_size=int(cfg.model.get("img_size", 224)),
        num_frames=num_frames,
        tubelet_t=int(cfg.model.get("tubelet_t", 2)),
        patch_size=int(cfg.model.get("patch_size", 16)),
        in_chans=int(cfg.model.get("in_chans", 3)),
        encoder_dim=int(cfg.model.get("encoder_dim", 768)),
        encoder_depth=int(cfg.model.get("encoder_depth", 12)),
        encoder_heads=int(cfg.model.get("encoder_heads", 12)),
        decoder_dim=int(cfg.model.get("decoder_dim", 384)),
        decoder_depth=int(cfg.model.get("decoder_depth", 4)),
        decoder_heads=int(cfg.model.get("decoder_heads", 6)),
        mlp_ratio=int(cfg.model.get("mlp_ratio", 4)),
        mask_ratio=float(cfg.model.get("mask_ratio", 0.75)),
        decoder_mask_ratio=float(cfg.model.get("decoder_mask_ratio", 0.5)),
        dropout=float(cfg.model.get("dropout", 0.0)),
        drop_path_rate=float(cfg.model.get("drop_path_rate", 0.0)),
        norm_pix_loss=bool(cfg.model.get("norm_pix_loss", True)),
    )


def cosine_lr(step: int, total_steps: int, warmup_steps: int,
              base_lr: float, min_lr: float = 1e-6) -> float:
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_distributed = world_size > 1
    is_main = (local_rank == 0)

    if is_distributed:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")

    if is_main:
        print(OmegaConf.to_yaml(cfg))
    set_seed(int(cfg.dataset.seed) + local_rank)

    # ── Data : on collecte tous les samples (labels non utilisés) ────────────
    train_dir = Path(cfg.dataset.train_dir).resolve()
    samples = collect_video_samples(train_dir)

    # Pretrain self-supervisé : on peut (et on devrait) inclure val_dir pour
    # avoir plus de données vidéo. Aucune étiquette n'est utilisée par MAE.
    include_val = bool(cfg.training.get("include_val_in_pretrain", False))
    if include_val:
        val_dir = Path(cfg.dataset.val_dir).resolve()
        val_samples = collect_video_samples(val_dir)
        n_train, n_val = len(samples), len(val_samples)
        samples = samples + val_samples
        if is_main:
            print(f"[data] include_val_in_pretrain=True : "
                  f"{n_train} train + {n_val} val = {len(samples)} clips total "
                  f"(+{100*n_val/max(n_train,1):.1f}%)")
    elif is_main:
        print(f"[data] include_val_in_pretrain=False : {len(samples)} clips (train_dir only)")

    transform = build_transforms(is_training=True, use_imagenet_norm=False)
    dataset = VideoFrameDataset(
        root_dir=train_dir,
        num_frames=int(cfg.dataset.num_frames),
        transform=transform,
        sample_list=samples,
        intra_class_swap_p=0.0,
    )

    if is_distributed:
        sampler = DistributedSampler(dataset, num_replicas=world_size,
                                     rank=local_rank, shuffle=True,
                                     seed=int(cfg.dataset.seed))
        loader = DataLoader(dataset, batch_size=int(cfg.training.batch_size),
                            sampler=sampler, num_workers=int(cfg.training.num_workers),
                            pin_memory=True, drop_last=True)
    else:
        sampler = None
        loader = DataLoader(dataset, batch_size=int(cfg.training.batch_size),
                            shuffle=True, num_workers=int(cfg.training.num_workers),
                            pin_memory=(device.type == "cuda"), drop_last=True)

    # ── Model ────────────────────────────────────────────────────────────────
    model = build_mae(cfg, int(cfg.dataset.num_frames)).to(device)
    if is_main:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"VideoMAEv2 | total params: {n_params/1e6:.2f}M | "
              f"num_patches: {model.patch_embed.num_patches} | "
              f"mask_ratio: {cfg.model.mask_ratio} | "
              f"decoder_mask_ratio: {cfg.model.decoder_mask_ratio}")

    # ── wandb (rank 0 only, opt-in) ──────────────────────────────────────────
    use_wandb = is_main and bool(cfg.training.get("wandb_project"))
    if use_wandb:
        import wandb
        wandb.init(
            project=str(cfg.training.wandb_project),
            name=str(cfg.training.get("wandb_run_name", f"mae_mask{cfg.model.mask_ratio}")),
            config=OmegaConf.to_container(cfg, resolve=True),
            resume="allow",
            id=str(cfg.training.get("wandb_run_id")) if cfg.training.get("wandb_run_id") else None,
        )
        wandb.summary["n_params_M"] = n_params / 1e6
        wandb.summary["num_patches"] = model.patch_embed.num_patches

    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # ── Optimizer (paper: AdamW betas=(0.9, 0.95)) ───────────────────────────
    base_lr = float(cfg.training.lr) * int(cfg.training.batch_size) * world_size / 256
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr,
                                  betas=(0.9, 0.95),
                                  weight_decay=float(cfg.training.get("weight_decay", 0.05)))

    # AMP : bf16 par défaut (RTX 4000 Ada / L40 / A100 supportent nativement).
    # bf16 a la même dynamique que fp32 -> pas d'overflow attention softmax,
    # pas besoin de GradScaler. fp16 est gardé en option (legacy).
    amp_dtype_str = str(cfg.training.get("amp_dtype", "bf16")).lower()
    if amp_dtype_str == "bf16" and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
        use_scaler = False
    elif amp_dtype_str == "fp32":
        amp_dtype = torch.float32
        use_scaler = False
    else:
        amp_dtype = torch.float16
        use_scaler = True
    scaler = GradScaler(device="cuda", enabled=use_scaler)
    if is_main:
        print(f"AMP dtype: {amp_dtype} | GradScaler: {use_scaler}")

    epochs = int(cfg.training.epochs)
    warmup_epochs = int(cfg.training.get("warmup_epochs", 10))
    steps_per_epoch = len(loader)
    total_steps = epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch

    checkpoint_path = Path(cfg.training.checkpoint_path).resolve()
    latest_path = checkpoint_path.with_name(checkpoint_path.stem + "_latest" + checkpoint_path.suffix)
    global_step = 0
    best_loss = float("inf")
    start_epoch = 0

    # ── Resume ───────────────────────────────────────────────────────────────
    # Deux modes :
    #   - resume normal : reprend optimizer + epoch + global_step (continue le run)
    #   - warm_restart=true : ne charge QUE les weights ; optimizer + schedule
    #     repartent à zéro -> permet de pousser un pretrain déjà convergé avec
    #     un nouveau cosine schedule (peak LR plus bas typiquement).
    resume_from = cfg.training.get("resume_from")
    warm_restart = bool(cfg.training.get("warm_restart", False))
    if resume_from:
        resume_path = Path(resume_from).resolve()
        ck = torch.load(resume_path, map_location="cpu", weights_only=False)
        get_module(model).load_state_dict(ck["model_state_dict"])
        if warm_restart:
            # Ne charge ni l'optimizer ni le scaler : nouveau schedule from scratch.
            # best_loss garde celui du checkpoint pour ne sauver que des progrès.
            best_loss = float(ck.get("best_loss", ck.get("loss", float("inf"))))
            if is_main:
                print(f"[warm-restart] Loaded weights ONLY from {resume_path} "
                      f"(prev epoch={ck.get('epoch', '?')}, prev best_loss={best_loss:.5f}). "
                      f"Optimizer + schedule réinitialisés.")
        else:
            if "optimizer_state_dict" in ck:
                optimizer.load_state_dict(ck["optimizer_state_dict"])
            if "scaler_state_dict" in ck:
                scaler.load_state_dict(ck["scaler_state_dict"])
            start_epoch = int(ck.get("epoch", 0))
            global_step = int(ck.get("global_step", start_epoch * steps_per_epoch))
            best_loss = float(ck.get("best_loss", ck.get("loss", float("inf"))))
            if is_main:
                print(f"Resumed from {resume_path} | start_epoch={start_epoch} | "
                      f"global_step={global_step} | best_loss={best_loss:.5f}")

    nan_skip_count = 0
    nan_consecutive = 0  # NaN/Inf skips d'affilée
    max_consecutive_nan = int(cfg.training.get("max_consecutive_nan", 200))
    for epoch in range(start_epoch, epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        running = 0.0
        n_seen = 0
        for video_batch, _ in loader:
            video_batch = video_batch.to(device, non_blocking=True)

            lr = cosine_lr(global_step, total_steps, warmup_steps, base_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            optimizer.zero_grad()
            with autocast(device_type="cuda", dtype=amp_dtype):
                loss, _, _ = model(video_batch)

            # Garde-fou #1 : loss NaN/Inf -> skip avant backward
            if not torch.isfinite(loss):
                nan_skip_count += 1
                nan_consecutive += 1
                if is_main and nan_skip_count <= 5:
                    print(f"  [warn] non-finite loss at step {global_step} — batch skipped")
                optimizer.zero_grad(set_to_none=True)  # paranoïa : on vide tout
                global_step += 1
                if nan_consecutive >= max_consecutive_nan:
                    raise RuntimeError(
                        f"Aborting : {nan_consecutive} non-finite losses d'affilée "
                        f"(step {global_step}, epoch {epoch+1}). Le run ne progresse plus."
                    )
                continue

            if use_scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            else:
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Garde-fou #2 : gradients NaN/Inf -> skip step (la loss était finie
            # mais bp a produit du nan, e.g. division par zéro dans norm_pix_loss).
            if not torch.isfinite(grad_norm):
                nan_skip_count += 1
                nan_consecutive += 1
                if is_main and nan_skip_count <= 5:
                    print(f"  [warn] non-finite grad_norm at step {global_step} — step skipped")
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if nan_consecutive >= max_consecutive_nan:
                    raise RuntimeError(
                        f"Aborting : {nan_consecutive} non-finite grad_norm d'affilée."
                    )
                continue

            # Tout est OK -> reset le compteur consécutif
            nan_consecutive = 0

            if use_scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            running += float(loss.item()) * video_batch.size(0)
            n_seen += video_batch.size(0)
            global_step += 1

            if use_wandb and global_step % 20 == 0:
                wandb.log({"train/loss_step": float(loss.item()),
                           "train/lr": lr,
                           "train/step": global_step,
                           "train/nan_skips": nan_skip_count})

        avg_loss = running / max(1, n_seen)
        if is_main:
            print(f"Ep {epoch+1:3d}/{epochs} | lr {lr:.2e} | recon loss {avg_loss:.5f}")
            if use_wandb:
                wandb.log({"train/loss_epoch": avg_loss,
                           "train/lr_epoch": lr,
                           "epoch": epoch + 1})

            # Payload complet (resume-friendly)
            payload = {
                "model_state_dict": get_module(model).state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "config": OmegaConf.to_container(cfg, resolve=True),
                "epoch": epoch + 1,
                "global_step": global_step,
                "loss": avg_loss,
                "best_loss": min(best_loss, avg_loss),
            }
            # 1) "latest" : pour reprendre d'où on en est
            torch.save(payload, latest_path)
            # 2) "best" : si la loss a baissé
            if avg_loss < best_loss:
                best_loss = avg_loss
                payload["best_loss"] = best_loss
                torch.save(payload, checkpoint_path)
                print(f"  Saved BEST to {checkpoint_path} (loss={avg_loss:.5f})")

    if use_wandb:
        wandb.finish()
    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
