"""
Train a video classifier on folders of frames.

Run from the ``src/`` directory (so ``configs/`` resolves)::
    python train.py
    python train.py experiment=cnn_lstm

Pick an **experiment** under ``configs/experiment/`` (each one selects a model and can
add more overrides). You can still override any key, e.g. ``model.pretrained=false``.

Training uses ``dataset.train_dir`` and ``split_train_val`` for an internal train/val
split; the dedicated ``dataset.val_dir`` is for ``evaluate.py`` only.

Multi-GPU (DDP) — launch with torchrun::
    torchrun --nproc_per_node=4 src/train.py experiment=model5_B_experiment \\
        'training.checkpoint_path=${hydra:runtime.cwd}/best_model_v6_8.pt' \\
        +training.resume_from=best_model_v6_7.pt \\
        training.batch_size=32
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import hydra
import torch
import torch.distributed as dist
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.amp import autocast, GradScaler

from dataset.video_dataset import VideoFrameDataset, collect_video_samples
from models.cnn_baseline import CNNBaseline
from models.cnn_lstm import CNNLSTM
from models.model1_A import VideoViT
from models.model12_A import VideoViT_CNN_2
from models.model2_A import VideoViT_CNN_3
from models.model3_A import R2Plus1D
from utils import build_transforms, set_seed, split_train_val
from mixup import MixupCutmix, SoftTargetCrossEntropy


def get_module(model: nn.Module) -> nn.Module:
    """Unwrap DistributedDataParallel to get the underlying module."""
    return model.module if isinstance(model, DDP) else model


def build_model(cfg: DictConfig) -> nn.Module:
    """Create the model described by cfg.model.name."""
    name = cfg.model.name
    num_classes = cfg.model.num_classes
    pretrained = cfg.model.get("pretrained", False)

    if name == "cnn_baseline":
        return CNNBaseline(num_classes=num_classes, pretrained=pretrained)
    if name == "cnn_lstm":
        hidden = cfg.model.get("lstm_hidden_size", 512)
        return CNNLSTM(
            num_classes=num_classes,
            pretrained=pretrained,
            lstm_hidden_size=int(hidden),
        )
    if name == "mvit_test_1":
        from models.model1 import MViT_test_1
        freeze_backbone = cfg.model.get("freeze_backbone", False)
        return MViT_test_1(num_classes=num_classes, pretrained=pretrained, freeze_backbone=freeze_backbone)


    if name == "model1.2_A":
        hidden_dim = int(cfg.model.get("hidden_dim", 512))
        n_heads = int(cfg.model.get("attention_heads", 8))
        dropout_rate = float(cfg.model.get("dropout_rate", 0.1))
        num_frames = int(cfg.dataset.num_frames)
        return VideoViT_CNN_2(num_classes=num_classes, hidden_dim=hidden_dim,
                        n_heads=n_heads, dropout_rate=dropout_rate, num_frames=num_frames, pretrained=pretrained)
    if name == "model2_A":
        hidden_dim = int(cfg.model.get("hidden_dim", 512))
        n_heads = int(cfg.model.get("attention_heads", 8))
        dropout_rate = float(cfg.model.get("dropout_rate", 0.1))
        num_frames = int(cfg.dataset.num_frames)
        return VideoViT_CNN_3(num_classes=num_classes, hidden_dim=hidden_dim,
                        n_heads=n_heads, dropout_rate=dropout_rate, num_frames=num_frames, pretrained=pretrained)
    if name == "model3_A":
        dropout3d_p = float(cfg.model.get("dropout3d_p", 0.3))
        drop_path_rate = float(cfg.model.get("drop_path_rate", 0.1))
        embed_dim = int(cfg.model.get("embed_dim", 512))
        n_heads = int(cfg.model.get("n_heads", 8))
        n_temporal_layers = int(cfg.model.get("n_temporal_layers", 1))
        transformer_dropout = float(cfg.model.get("transformer_dropout", 0.3))
        transformer_drop_path_rate = float(cfg.model.get("transformer_drop_path_rate", 0.1))
        head_dropout = float(cfg.model.get("head_dropout", 0.5))
        num_frames = int(cfg.dataset.num_frames)

        return R2Plus1D(
            num_classes=num_classes,
            num_frames=num_frames,
            dropout3d_p=dropout3d_p,
            drop_path_rate=drop_path_rate,
            embed_dim=embed_dim,
            n_heads=n_heads,
            n_temporal_layers=n_temporal_layers,
            transformer_dropout=transformer_dropout,
            transformer_drop_path_rate=transformer_drop_path_rate,
            head_dropout=head_dropout,
            pretrained=pretrained,
        )

    if name == "model4_A":
        from models.model4_A import CompactVideoModel
        embed_dim = int(cfg.model.get("embed_dim", 256))
        n_heads = int(cfg.model.get("n_heads", 4))
        n_temporal_layers = int(cfg.model.get("n_temporal_layers", 1))
        dropout_rate = float(cfg.model.get("dropout_rate", 0.2))
        drop_path_rate = float(cfg.model.get("drop_path_rate", 0.1))
        head_dropout = float(cfg.model.get("head_dropout", 0.4))
        return CompactVideoModel(
            num_classes=num_classes,
            num_frames=int(cfg.dataset.num_frames),
            embed_dim=embed_dim,
            n_heads=n_heads,
            n_temporal_layers=n_temporal_layers,
            dropout_rate=dropout_rate,
            drop_path_rate=drop_path_rate,
            head_dropout=head_dropout,
            pretrained=pretrained,
        )

    if name == "model5_A":
        from models.model5_A import R2Plus1D_TransformerHead
        embed_dim = int(cfg.model.get("embed_dim", 512))
        n_heads = int(cfg.model.get("n_heads", 8))
        n_temporal_layers = int(cfg.model.get("n_temporal_layers", 1))
        transformer_dropout = float(cfg.model.get("transformer_dropout", 0.3))
        transformer_drop_path_rate = float(cfg.model.get("transformer_drop_path_rate", 0.1))
        head_dropout = float(cfg.model.get("head_dropout", 0.5))
        dropout3d_p = float(cfg.model.get("dropout3d_p", 0.3))
        drop_path_rate = float(cfg.model.get("drop_path_rate", 0.1))

        return R2Plus1D_TransformerHead(
            num_classes=num_classes,
            num_frames=int(cfg.dataset.num_frames),
            dropout3d_p=dropout3d_p,
            drop_path_rate=drop_path_rate,
            embed_dim=embed_dim,
            n_heads=n_heads,
            n_temporal_layers=n_temporal_layers,
            transformer_dropout=transformer_dropout,
            transformer_drop_path_rate=transformer_drop_path_rate,
            head_dropout=head_dropout,
            pretrained=pretrained,
        )

    if name == "model5_B":
        from models.model5_B import R2Plus1D_SpatioTemporalTransformer
        return R2Plus1D_SpatioTemporalTransformer(
            num_classes=num_classes,
            num_frames=int(cfg.dataset.num_frames),
            tokens_per_frame=int(cfg.model.get("tokens_per_frame", 49)),
            dropout3d_p=float(cfg.model.get("dropout3d_p", 0.1)),
            drop_path_rate=float(cfg.model.get("drop_path_rate", 0.05)),
            embed_dim=int(cfg.model.get("embed_dim", 512)),
            n_heads=int(cfg.model.get("n_heads", 8)),
            n_temporal_layers=int(cfg.model.get("n_temporal_layers", 2)),
            transformer_dropout=float(cfg.model.get("transformer_dropout", 0.1)),
            transformer_drop_path_rate=float(cfg.model.get("transformer_drop_path_rate", 0.1)),
            head_dropout=float(cfg.model.get("head_dropout", 0.3)),
            use_tsm=bool(cfg.model.get("use_tsm", True)),
            tsm_fold_div=int(cfg.model.get("tsm_fold_div", 8)),
            pretrained=pretrained,
        )

    if name == "model7_A":
        from models.model7_A import VideoViT
        return VideoViT(
            num_classes=num_classes,
            num_frames=int(cfg.dataset.num_frames),
            img_size=int(cfg.model.get("img_size", 224)),
            tubelet_t=int(cfg.model.get("tubelet_t", 2)),
            patch_size=int(cfg.model.get("patch_size", 16)),
            in_chans=int(cfg.model.get("in_chans", 3)),
            embed_dim=int(cfg.model.get("embed_dim", 768)),
            depth=int(cfg.model.get("depth", 12)),
            n_heads=int(cfg.model.get("n_heads", 12)),
            mlp_ratio=int(cfg.model.get("mlp_ratio", 4)),
            dropout=float(cfg.model.get("dropout", 0.0)),
            drop_path_rate=float(cfg.model.get("drop_path_rate", 0.2)),
            head_dropout=float(cfg.model.get("head_dropout", 0.0)),
            pool=str(cfg.model.get("pool", "avg")),
            pretrained=pretrained,
        )

    if name == "model6_A":
        from models.model6_A import VideoMaxViT
        return VideoMaxViT(
            num_classes=num_classes,
            num_frames=int(cfg.dataset.num_frames),
            stem_channels=int(cfg.model.get("stem_channels", 64)),
            stage_channels=tuple(cfg.model.get("stage_channels", [128, 256, 384])),
            num_blocks=tuple(cfg.model.get("num_blocks", [2, 2, 2])),
            n_heads=tuple(cfg.model.get("n_heads", [4, 8, 8])),
            strides=tuple(cfg.model.get("strides", [2, 2, 2])),
            window_size=int(cfg.model.get("window_size", 7)),
            dropout=float(cfg.model.get("dropout", 0.1)),
            drop_path_rate=float(cfg.model.get("drop_path_rate", 0.15)),
            head_dropout=float(cfg.model.get("head_dropout", 0.3)),
            expand=int(cfg.model.get("expand", 4)),
            se_ratio=float(cfg.model.get("se_ratio", 0.25)),
            mlp_ratio=int(cfg.model.get("mlp_ratio", 4)),
            pretrained=pretrained,
        )

    raise ValueError(f"Unknown model.name: {name}")


# ─── Layer-wise LR decay (LLDR) ──────────────────────────────────────────────
def _videovit_layer_id(name: str, depth: int) -> int:
    """
    Mappe un nom de paramètre d'un VideoViT (model7_A) -> layer_id [0 .. depth+1].
    Convention MAE :
      layer 0          : patch_embed, cls_token (features bas-niveau)
      layer i+1        : blocks.{i}.*
      layer depth+1    : norm.*, head.*  (tête + LN final, full LR)
    """
    if name.startswith("patch_embed") or name == "cls_token" or name.startswith("cls_token"):
        return 0
    if name.startswith("blocks."):
        # blocks.{i}.{...}
        try:
            block_idx = int(name.split(".")[1])
        except (IndexError, ValueError):
            return depth + 1
        return block_idx + 1
    # norm.*, head.*, head_dropout, et toute couche non identifiée -> full LR
    return depth + 1


def build_lldr_param_groups(
    model: nn.Module,
    base_lr: float,
    layer_decay: float,
    weight_decay: float,
    depth: int,
) -> List[Dict[str, Any]]:
    """
    Construit les param groups pour AdamW avec layer-wise LR decay.

    Decay : LR au layer L = base_lr * layer_decay ** (num_layers - L)
            avec num_layers = depth + 1.
        -> layer 0 (patch_embed) : LR ~ base_lr * decay^(depth+1)    (très petit)
        -> layer depth+1 (head)  : LR ~ base_lr                      (full)

    Pas de weight decay sur les biases et tenseurs 1D (LayerNorm, etc.).
    """
    num_layers = depth + 1
    layer_scales = [layer_decay ** (num_layers - i) for i in range(num_layers + 1)]

    # key: (layer_id, no_wd) -> group dict
    groups: Dict[Tuple[int, bool], Dict[str, Any]] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        layer_id = _videovit_layer_id(name, depth)
        no_wd = (param.ndim <= 1) or name.endswith(".bias")
        key = (layer_id, no_wd)
        if key not in groups:
            scale = layer_scales[layer_id]
            groups[key] = {
                "params": [],
                "lr": base_lr * scale,
                "lr_scale": scale,
                "weight_decay": 0.0 if no_wd else weight_decay,
                "layer_id": layer_id,
                "group_name": f"layer{layer_id}_{'nowd' if no_wd else 'wd'}",
            }
        groups[key]["params"].append(param)

    # Tri par layer_id croissant pour que param_groups[0] = couche la plus basse (lr min).
    return sorted(groups.values(), key=lambda g: (g["layer_id"], g["group_name"]))


def make_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
    total_epochs: int,
    min_lr_ratio: float = 0.01,
) -> LambdaLR:
    """LR multiplicatif : warmup linéaire 0->1 puis cosine 1->min_lr_ratio."""
    def _lr_lambda(epoch: int) -> float:
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        progress = min(max(progress, 0.0), 1.0)
        return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda=_lr_lambda)


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    mixer
) -> Tuple[float, float]:
    """Returns (average loss, top-1 accuracy) on the training set for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for video_batch, labels in data_loader:
        # video_batch: (B, T, C, H, W), labels: (B,)
        video_batch = video_batch.to(device)
        labels = labels.to(device)

        # Mix en FP32 sur device, avant autocast.
        mixed_x, soft_targets = mixer(video_batch, labels)
        optimizer.zero_grad()

        with autocast(device_type="cuda"):
            logits = model(mixed_x)  # (B, num_classes)
            loss = loss_fn(logits, soft_targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()

        running_loss += float(loss.item()) * labels.size(0)
        predictions = logits.argmax(dim=1)
        correct += int((predictions == labels).sum().item())
        total += labels.size(0)

    average_loss = running_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return average_loss, accuracy


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Returns (average loss, top-1 accuracy) on the validation loader."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for video_batch, labels in data_loader:
        video_batch = video_batch.to(device)
        labels = labels.to(device)

        logits = model(video_batch)
        loss = loss_fn(logits, labels)

        running_loss += float(loss.item()) * labels.size(0)
        predictions = logits.argmax(dim=1)
        correct += int((predictions == labels).sum().item())
        total += labels.size(0)

    average_loss = running_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return average_loss, accuracy


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:

    # ── DDP setup ──────────────────────────────────────────────────────────────
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_distributed = world_size > 1
    is_main = (local_rank == 0)

    if is_distributed:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device_str = cfg.training.device
        if device_str == "cuda" and not torch.cuda.is_available():
            if is_main:
                print("CUDA not available; using CPU.")
            device_str = "cpu"
        device = torch.device(device_str)

    if is_main:
        print(OmegaConf.to_yaml(cfg))

    set_seed(int(cfg.dataset.seed) + local_rank)

    # ── Data ───────────────────────────────────────────────────────────────────
    train_dir = Path(cfg.dataset.train_dir).resolve()
    all_samples = collect_video_samples(train_dir)

    max_samples = cfg.dataset.get("max_samples")
    if max_samples is not None:
        all_samples = all_samples[: int(max_samples)]

    train_samples, val_samples = split_train_val(
        all_samples,
        val_ratio=float(cfg.dataset.val_ratio),
        seed=int(cfg.dataset.seed),
    )

    use_imagenet_norm = bool(cfg.model.get("pretrained", False))
    train_transform = build_transforms(is_training=True, use_imagenet_norm=use_imagenet_norm)
    eval_transform  = build_transforms(is_training=False, use_imagenet_norm=use_imagenet_norm)

    train_dataset = VideoFrameDataset(
        root_dir=train_dir,
        num_frames=int(cfg.dataset.num_frames),
        transform=train_transform,
        sample_list=train_samples,
        intra_class_swap_p=0.2,
    )
    val_dataset = VideoFrameDataset(
        root_dir=train_dir,
        num_frames=int(cfg.dataset.num_frames),
        transform=eval_transform,
        sample_list=val_samples,
        intra_class_swap_p=0.2,
    )

    # Train loader: DistributedSampler in DDP mode, else shuffle=True
    if is_distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=local_rank,
            shuffle=True,
            seed=int(cfg.dataset.seed),
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=int(cfg.training.batch_size),
            sampler=train_sampler,
            num_workers=int(cfg.training.num_workers),
            pin_memory=True,
        )
    else:
        train_sampler = None
        train_loader = DataLoader(
            train_dataset,
            batch_size=int(cfg.training.batch_size),
            shuffle=True,
            num_workers=int(cfg.training.num_workers),
            pin_memory=(device.type == "cuda"),
        )

    # Val loaders: only rank 0 evaluates
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(cfg.training.batch_size),
        shuffle=False,
        num_workers=int(cfg.training.num_workers),
        pin_memory=(device.type == "cuda"),
    )

    real_val_dir = Path(cfg.dataset.val_dir).resolve()
    real_val_samples = collect_video_samples(real_val_dir)
    real_val_dataset = VideoFrameDataset(
        root_dir=real_val_dir,
        num_frames=int(cfg.dataset.num_frames),
        transform=eval_transform,
        sample_list=real_val_samples,
        intra_class_swap_p=0.0,
    )
    real_val_loader = DataLoader(
        real_val_dataset,
        batch_size=int(cfg.training.batch_size),
        shuffle=False,
        num_workers=int(cfg.training.num_workers),
        pin_memory=(device.type == "cuda"),
    )

    # ── Model ──────────────────────────────────────────────────────────────────
    model = build_model(cfg).to(device)

    freeze_epochs = int(cfg.training.get("freeze_backbone_epochs", 0))
    best_val_accuracy = 0.0

    resume_path = cfg.training.get("resume_from")
    resumed = False
    if resume_path:
        resume_path = Path(resume_path).resolve()
        # All ranks load the same checkpoint; DDP will sync anyway.
        raw = torch.load(resume_path, map_location=device)
        model.load_state_dict(raw["model_state_dict"])
        best_val_accuracy = float(raw.get("real_val_accuracy", raw.get("val_accuracy", 0.0)))
        if is_main:
            print(f"Resumed from {resume_path} (val acc={best_val_accuracy:.4f})")
        resumed = True
    else:
        backbone_init = cfg.training.get("backbone_init_from")
        if backbone_init:
            get_module(model).load_backbone_from_model3(Path(backbone_init).resolve())

        # Init depuis un pretrain VideoMAE V2 :
        #   - model7_A : load 100 % de l'encoder (chemin direct).
        #   - model6_A : transfert sélectif des 4 derniers blocs vers stage 3.
        videomae_init = cfg.training.get("videomae_pretrain_from")
        if videomae_init:
            videomae_path = Path(videomae_init).resolve()
            if cfg.model.name == "model7_A":
                get_module(model).load_from_videomae_v2(videomae_path)
                if is_main:
                    print(f"model7_A : encoder pretrain rechargé à 100 % depuis {videomae_path}")
            elif cfg.model.name == "model6_A":
                from models.videomae_v2 import VideoMAEv2
                ck = torch.load(videomae_path, map_location="cpu")
                mae_cfg = ck.get("config", {}).get("model", {})
                mae = VideoMAEv2(
                    img_size=int(mae_cfg.get("img_size", 224)),
                    num_frames=int(cfg.dataset.num_frames),
                    tubelet_t=int(mae_cfg.get("tubelet_t", 2)),
                    patch_size=int(mae_cfg.get("patch_size", 16)),
                    in_chans=int(mae_cfg.get("in_chans", 3)),
                    encoder_dim=int(mae_cfg.get("encoder_dim", 768)),
                    encoder_depth=int(mae_cfg.get("encoder_depth", 12)),
                    encoder_heads=int(mae_cfg.get("encoder_heads", 12)),
                    decoder_dim=int(mae_cfg.get("decoder_dim", 384)),
                    decoder_depth=int(mae_cfg.get("decoder_depth", 4)),
                    decoder_heads=int(mae_cfg.get("decoder_heads", 6)),
                    mlp_ratio=int(mae_cfg.get("mlp_ratio", 4)),
                    mask_ratio=float(mae_cfg.get("mask_ratio", 0.75)),
                    decoder_mask_ratio=float(mae_cfg.get("decoder_mask_ratio", 0.5)),
                    norm_pix_loss=bool(mae_cfg.get("norm_pix_loss", True)),
                )
                mae.load_state_dict(ck["model_state_dict"])
                num_xfer = int(cfg.training.get("videomae_num_blocks_to_transfer", 4))
                mae.transfer_to_model6_stage3(get_module(model), num_blocks_to_transfer=num_xfer)
                if is_main:
                    print(f"Initialisé stage3 de model6 depuis pretrain VideoMAEv2 : {videomae_path}")
            else:
                raise ValueError(
                    f"videomae_pretrain_from non supporté pour {cfg.model.name} "
                    "(seulement model6_A et model7_A pour l'instant)."
                )

    # Wrap with DDP after loading weights (DDP broadcasts rank-0 params to all ranks)
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    def _count_params(m):
        m = get_module(m)
        total = sum(p.numel() for p in m.parameters())
        trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
        return total, trainable

    if is_main:
        total_p, trainable_p = _count_params(model)
        print(f"Model: {cfg.model.name} | total params: {total_p/1e6:.2f}M | "
              f"trainable: {trainable_p/1e6:.2f}M ({100*trainable_p/max(total_p,1):.1f}%)")
        print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)} | "
              f"Batch/GPU: {cfg.training.batch_size} | Effective batch: {int(cfg.training.batch_size)*world_size} | "
              f"GPUs: {world_size} | Epochs: {cfg.training.epochs} | Freeze backbone: {freeze_epochs} epochs")
        print("-" * 100)

    # ── wandb (rank 0 only, opt-in via cfg.training.wandb_project) ──────────
    use_wandb = is_main and bool(cfg.training.get("wandb_project"))
    if use_wandb:
        import wandb
        wandb.init(
            project=str(cfg.training.wandb_project),
            name=str(cfg.training.get("wandb_run_name", cfg.model.name)),
            config=OmegaConf.to_container(cfg, resolve=True),
            resume="allow",
            id=str(cfg.training.get("wandb_run_id")) if cfg.training.get("wandb_run_id") else None,
        )
        wandb.summary["total_params_M"] = total_p / 1e6
        wandb.summary["trainable_params_M"] = trainable_p / 1e6

    prev_val_top1 = None

    mixer = MixupCutmix(
        num_classes=int(cfg.model.num_classes),
        mixup_alpha=float(cfg.training.get("mixup_alpha", 0.2)),
        cutmix_alpha=float(cfg.training.get("cutmix_alpha", 1.0)),
        mixup_prob=float(cfg.training.get("mixup_prob", 0.5)),
        cutmix_prob=float(cfg.training.get("cutmix_prob", 0.5)),
        label_smoothing=float(cfg.training.get("label_smoothing", 0.1)),
    )

    train_loss_fn = SoftTargetCrossEntropy()
    val_loss_fn   = nn.CrossEntropyLoss(label_smoothing=0.1)

    # ── Optimizer / scheduler (LLDR + warmup si layer_decay défini) ────────
    weight_decay = float(cfg.training.get("weight_decay", 1e-3))
    layer_decay  = cfg.training.get("layer_decay")
    warmup_epochs = int(cfg.training.get("warmup_epochs", 0))
    min_lr_ratio = float(cfg.training.get("min_lr_ratio", 0.01))
    total_epochs = int(cfg.training.epochs)

    if layer_decay is not None:
        if cfg.model.name != "model7_A":
            raise ValueError(
                f"layer_decay n'est implémenté que pour model7_A (VideoViT) "
                f"pour l'instant ; reçu model.name={cfg.model.name}."
            )
        if freeze_epochs > 0 and is_main:
            print("[warn] LLDR activé : freeze_backbone_epochs ignoré "
                  "(les couches basses ont déjà un LR très faible).")
        depth = int(cfg.model.get("depth", 12))
        adam_betas = tuple(cfg.training.get("betas", [0.9, 0.95]))  # MAE-recommended
        param_groups = build_lldr_param_groups(
            get_module(model),
            base_lr=float(cfg.training.lr),
            layer_decay=float(layer_decay),
            weight_decay=weight_decay,
            depth=depth,
        )
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=float(cfg.training.lr),  # ignoré par groupe puisque chaque groupe a son lr
            betas=adam_betas,
        )
        if is_main:
            lrs = [g["lr"] for g in param_groups]
            print(f"LLDR ON | layer_decay={layer_decay} | depth={depth} | "
                  f"{len(param_groups)} param groups | "
                  f"LR layer0 (min)={min(lrs):.2e} -> LR head (max)={max(lrs):.2e} | "
                  f"warmup_epochs={warmup_epochs} | weight_decay={weight_decay} | "
                  f"betas={adam_betas}")
    else:
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=float(cfg.training.lr),
            weight_decay=weight_decay,
        )

    if layer_decay is not None or warmup_epochs > 0:
        scheduler = make_warmup_cosine_scheduler(
            optimizer,
            warmup_epochs=warmup_epochs,
            total_epochs=total_epochs,
            min_lr_ratio=min_lr_ratio,
        )
    else:
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_epochs,
            eta_min=float(cfg.training.lr) * min_lr_ratio,
        )
    checkpoint_path = Path(cfg.training.checkpoint_path).resolve()

    scaler = GradScaler(device="cuda")

    for epoch in range(int(cfg.training.epochs)):

        # ── Freeze / unfreeze backbone ─────────────────────────────────────────
        if epoch == 0 and freeze_epochs > 0 and not resumed:
            get_module(model).set_backbone_trainable(False)
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(trainable_params, lr=float(cfg.training.lr) * 10,
                                          weight_decay=1e-3)
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=max(1, freeze_epochs),
                eta_min=float(cfg.training.lr) * 1.0,
            )
            if is_main:
                _, trainable_p = _count_params(model)
                print(f">> Phase 1 (freeze backbone) | trainable: {trainable_p/1e6:.2f}M | "
                      f"lr: {float(cfg.training.lr)*10:.2e}")

        elif epoch == freeze_epochs and freeze_epochs > 0 and not resumed:
            get_module(model).set_backbone_trainable(True)
            optimizer = torch.optim.AdamW(model.parameters(),
                                          lr=float(cfg.training.lr), weight_decay=1e-3)
            scheduler = CosineAnnealingLR(optimizer,
                T_max=int(cfg.training.epochs) - freeze_epochs,
                eta_min=float(cfg.training.lr) * 0.01)
            if is_main:
                _, trainable_p = _count_params(model)
                print(f">> Phase 2 (unfreeze all)  | trainable: {trainable_p/1e6:.2f}M | "
                      f"lr: {float(cfg.training.lr):.2e}")

        current_lr = max(g["lr"] for g in optimizer.param_groups)

        # ── Set epoch for DistributedSampler (reshuffles each epoch) ──────────
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # ── Training step (all ranks) ─────────────────────────────────────────
        train_loss, train_acc = train_one_epoch(
            model, train_loader, train_loss_fn, optimizer, scaler, device, mixer
        )

        # ── Evaluation (rank 0 only) ──────────────────────────────────────────
        if is_main:
            val_loss, val_acc = evaluate_epoch(model, val_loader, val_loss_fn, device)
            real_val_loss, real_val_acc = evaluate_epoch(model, real_val_loader, val_loss_fn, device)
        else:
            val_acc, real_val_acc = 0.0, 0.0

        # Broadcast real_val_acc from rank 0 to all ranks (needed for consistent scheduler/save)
        if is_distributed:
            _acc_tensor = torch.tensor(real_val_acc, device=device)
            dist.broadcast(_acc_tensor, src=0)
            real_val_acc = float(_acc_tensor.item())

        # ── Clean train acc every 3 epochs (rank 0 only) ─────────────────────
        if is_main and (epoch + 1) % 3 == 0:
            train_dataset.transform = eval_transform
            _, clean_train_acc = evaluate_epoch(model, train_loader, val_loss_fn, device)
            train_dataset.transform = train_transform
            print(f"  clean train acc: {clean_train_acc:.4f}")

        scheduler.step()
        current_lr = max(g["lr"] for g in optimizer.param_groups)

        delta_val = (real_val_acc - prev_val_top1) if prev_val_top1 is not None else 0.0
        prev_val_top1 = real_val_acc

        if is_main:
            print(
                f"Ep {epoch + 1:3d}/{cfg.training.epochs} | "
                f"lr {current_lr:.2e} | "
                f"train top1 {train_acc:.4f} | "
                f"int val top1 {val_acc:.4f} | "
                f"REAL val top1 {real_val_acc:.4f} | "
                f"gap {val_acc - real_val_acc:+.3f} | "
                f"dv {delta_val:+.3f}"
            )
            if use_wandb:
                log_dict = {
                    "epoch": epoch + 1,
                    "train/lr": current_lr,
                    "train/loss": train_loss,
                    "train/top1": train_acc,
                    "val/loss": val_loss,
                    "val/top1": val_acc,
                    "real_val/loss": real_val_loss,
                    "real_val/top1": real_val_acc,
                    "val/gap_int_minus_real": val_acc - real_val_acc,
                    "val/delta_real_top1": delta_val,
                }
                if (epoch + 1) % 3 == 0:
                    log_dict["train/clean_top1"] = clean_train_acc
                wandb.log(log_dict)

        # ── Checkpoint (rank 0 only) ──────────────────────────────────────────
        if is_main and real_val_acc > best_val_accuracy:
            best_val_accuracy = real_val_acc
            payload: Dict[str, Any] = {
                "model_state_dict": get_module(model).state_dict(),
                "model_name": cfg.model.name,
                "num_classes": int(cfg.model.num_classes),
                "pretrained": bool(cfg.model.get("pretrained", False)),
                "num_frames": int(cfg.dataset.num_frames),
                "val_accuracy": real_val_acc,
                "real_val_accuracy": real_val_acc,
                "config": OmegaConf.to_container(cfg, resolve=True),
            }
            if cfg.model.name == "cnn_lstm":
                payload["lstm_hidden_size"] = int(cfg.model.get("lstm_hidden_size", 512))

            torch.save(payload, checkpoint_path)
            print(f"  Saved new best model to {checkpoint_path} (real val acc={real_val_acc:.4f})")
            if use_wandb:
                wandb.summary["best_real_val_top1"] = best_val_accuracy

    if is_main:
        print(f"Done. Best validation accuracy: {best_val_accuracy:.4f}")
    if use_wandb:
        wandb.finish()

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
