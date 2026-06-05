"""
model6_A — VideoMaxViT (from scratch, 3D-natif)

Diffère explicitement de model5 (CNN + attention temporelle empilée à la fin) :
ici conv et attention sont ENTRELACÉES à chaque profondeur.

Chaque MaxViT block enchaîne :
  - MBConv3D : 1x1 expand -> 3x3x3 depthwise -> SE -> 1x1 project, résiduel.
               Mixe spatial + temporel localement, avec un coût faible.
  - Block (window) attention : self-attention dans des fenêtres locales
    (T, ws, ws). Attention locale dense (équivalent video de Swin sans shift).
  - Grid attention : self-attention sur une grille dilatée (ws*ws cellules,
    chaque cellule contient les tokens espacés par bh = H/ws). Donne une
    attention globale sparse en O(N) au lieu de O(N^2).

Stages (entrée 224x224, T=4) :
  stem (/4, ch=64)         -> (B, 64,  4, 56, 56)
  stage1 (stride=2)        -> (B, 128, 4, 28, 28)
  stage2 (stride=2)        -> (B, 256, 4, 14, 14)
  stage3 (stride=2)        -> (B, 384, 4,  7,  7)
  global avg pool -> LN -> Dropout -> Linear(384, 33)

Avec ws=7, à chaque stage on a soit plusieurs fenêtres (stages 1,2) soit
une seule (stage 3, équivalent à full self-attention). Tokens par fenêtre
de block-attn = T * ws * ws = 196.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ─── DropPath ────────────────────────────────────────────────────────────────
class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.floor(torch.rand(shape, dtype=x.dtype, device=x.device) + keep)
        return x * mask / keep


# ─── Briques transformer ─────────────────────────────────────────────────────
class FFN(nn.Module):
    def __init__(self, dim, dropout=0.0, expansion=4):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * expansion)
        self.fc2 = nn.Linear(dim * expansion, dim)
        self.drop = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x):
        return self.fc2(self.drop(self.act(self.fc1(x))))


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, n_heads, attn_dropout=0.0, proj_dropout=0.0):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj_drop = nn.Dropout(proj_dropout)

    def forward(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = self.attn_drop(attn.softmax(dim=-1))
        out = (attn @ v).transpose(1, 2).contiguous().view(B, N, D)
        return self.proj_drop(self.proj(out))


# ─── Window / Grid partition (3D) ────────────────────────────────────────────
def window_partition(x, ws):
    """(B, C, T, H, W) -> (B*nWh*nWw, T*ws*ws, C). H et W doivent être multiples de ws."""
    B, C, T, H, W = x.shape
    assert H % ws == 0 and W % ws == 0, f"H={H}, W={W} non divisibles par ws={ws}"
    nh, nw = H // ws, W // ws
    x = x.reshape(B, C, T, nh, ws, nw, ws)
    x = x.permute(0, 3, 5, 2, 4, 6, 1).contiguous()  # (B, nh, nw, T, ws, ws, C)
    return x.reshape(B * nh * nw, T * ws * ws, C)


def window_reverse(tokens, ws, T, H, W, C):
    nh, nw = H // ws, W // ws
    B = tokens.shape[0] // (nh * nw)
    x = tokens.reshape(B, nh, nw, T, ws, ws, C)
    x = x.permute(0, 6, 3, 1, 4, 2, 5).contiguous()  # (B, C, T, nh, ws, nw, ws)
    return x.reshape(B, C, T, H, W)


def grid_partition(x, gs):
    """
    Grille dilatée : gs*gs cellules, chacune contient les tokens espacés
    (stride bh, bw). Donne une attention globale sparse.
    (B, C, T, H, W) -> (B*gs*gs, T*bh*bw, C).
    """
    B, C, T, H, W = x.shape
    bh, bw = H // gs, W // gs
    assert bh * gs == H and bw * gs == W, f"H={H}, W={W} non divisibles par gs={gs}"
    x = x.reshape(B, C, T, bh, gs, bw, gs)
    x = x.permute(0, 4, 6, 2, 3, 5, 1).contiguous()  # (B, gs, gs, T, bh, bw, C)
    return x.reshape(B * gs * gs, T * bh * bw, C)


def grid_reverse(tokens, gs, T, H, W, C):
    bh, bw = H // gs, W // gs
    B = tokens.shape[0] // (gs * gs)
    x = tokens.reshape(B, gs, gs, T, bh, bw, C)
    x = x.permute(0, 6, 3, 4, 1, 5, 2).contiguous()  # (B, C, T, bh, gs, bw, gs)
    return x.reshape(B, C, T, H, W)


# ─── MBConv3D (avec Squeeze-and-Excitation) ──────────────────────────────────
class MBConv3D(nn.Module):
    def __init__(self, in_c, out_c, stride=1, expand=4, se_ratio=0.25, drop_path=0.0):
        super().__init__()
        mid = in_c * expand
        self.same_shape = (in_c == out_c) and (stride == 1)

        # Pré-BN
        self.bn0 = nn.BatchNorm3d(in_c)
        # Expansion 1x1
        self.expand = nn.Conv3d(in_c, mid, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(mid)
        # Depthwise 3x3x3 (mélange spatial + temporel)
        self.dw = nn.Conv3d(mid, mid, kernel_size=3,
                            stride=(1, stride, stride),
                            padding=1, groups=mid, bias=False)
        self.bn2 = nn.BatchNorm3d(mid)
        # Squeeze-and-Excitation (global sur T,H,W)
        se_dim = max(4, int(in_c * se_ratio))
        self.se_pool = nn.AdaptiveAvgPool3d(1)
        self.se_reduce = nn.Conv3d(mid, se_dim, 1)
        self.se_expand = nn.Conv3d(se_dim, mid, 1)
        # Projection 1x1
        self.project = nn.Conv3d(mid, out_c, kernel_size=1, bias=False)

        if self.same_shape:
            self.shortcut = nn.Identity()
        else:
            layers = []
            if stride > 1:
                layers.append(nn.AvgPool3d(kernel_size=(1, stride, stride),
                                           stride=(1, stride, stride)))
            layers.append(nn.Conv3d(in_c, out_c, kernel_size=1, bias=False))
            self.shortcut = nn.Sequential(*layers)

        self.act = nn.GELU()
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        shortcut = self.shortcut(x)
        out = self.act(self.bn1(self.expand(self.bn0(x))))
        out = self.act(self.bn2(self.dw(out)))
        # SE
        se = self.se_pool(out)
        se = torch.sigmoid(self.se_expand(self.act(self.se_reduce(se))))
        out = out * se
        out = self.project(out)
        return shortcut + self.drop_path(out)


# ─── Window/Grid attention block (Pre-Norm) ──────────────────────────────────
class PartitionAttention(nn.Module):
    def __init__(self, dim, n_heads, partition_size, mode,
                 dropout=0.0, drop_path=0.0, mlp_ratio=4):
        super().__init__()
        assert mode in ("block", "grid")
        self.mode = mode
        self.ws = partition_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, n_heads,
                                           attn_dropout=0.0, proj_dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = FFN(dim, dropout=dropout, expansion=mlp_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        # x : (B, C, T, H, W)
        B, C, T, H, W = x.shape
        if self.mode == "block":
            tokens = window_partition(x, self.ws)
        else:
            tokens = grid_partition(x, self.ws)
        tokens = tokens + self.drop_path(self.attn(self.norm1(tokens)))
        tokens = tokens + self.drop_path(self.mlp(self.norm2(tokens)))
        if self.mode == "block":
            return window_reverse(tokens, self.ws, T, H, W, C)
        return grid_reverse(tokens, self.ws, T, H, W, C)


class MaxViTBlock(nn.Module):
    """MBConv -> Block-Attn -> Grid-Attn (la signature MaxViT)."""
    def __init__(self, in_c, out_c, n_heads, window_size, stride=1,
                 dropout=0.0, drop_path=0.0, mlp_ratio=4,
                 expand=4, se_ratio=0.25):
        super().__init__()
        self.mbconv = MBConv3D(in_c, out_c, stride=stride, expand=expand,
                               se_ratio=se_ratio, drop_path=drop_path)
        self.block_attn = PartitionAttention(out_c, n_heads, window_size, "block",
                                             dropout=dropout, drop_path=drop_path,
                                             mlp_ratio=mlp_ratio)
        self.grid_attn = PartitionAttention(out_c, n_heads, window_size, "grid",
                                            dropout=dropout, drop_path=drop_path,
                                            mlp_ratio=mlp_ratio)

    def forward(self, x):
        x = self.mbconv(x)
        x = self.block_attn(x)
        x = self.grid_attn(x)
        return x


# ─── Modèle complet ──────────────────────────────────────────────────────────
class VideoMaxViT(nn.Module):
    """
    Args:
        num_classes        : 33
        num_frames         : T (= 4)
        stem_channels      : sortie du stem (64)
        stage_channels     : (128, 256, 384)
        num_blocks         : (2, 2, 2)
        n_heads            : (4, 8, 8) (un par stage)
        strides            : (2, 2, 2) — downsample spatial au 1er bloc du stage
        window_size        : 7
        dropout            : dropout dans attn/ffn
        drop_path_rate     : stochastic depth max (linéaire jusqu'à cette valeur)
        head_dropout       : dropout devant la fc finale
        expand             : facteur d'expansion MBConv
        se_ratio           : ratio du squeeze-and-excitation
        mlp_ratio          : ratio du FFN transformer
        pretrained         : ignoré (from scratch)
    """
    def __init__(self,
                 num_classes: int = 33,
                 num_frames: int = 4,
                 stem_channels: int = 64,
                 stage_channels=(128, 256, 384),
                 num_blocks=(2, 2, 2),
                 n_heads=(4, 8, 8),
                 strides=(2, 2, 2),
                 window_size: int = 7,
                 dropout: float = 0.1,
                 drop_path_rate: float = 0.1,
                 head_dropout: float = 0.3,
                 expand: int = 4,
                 se_ratio: float = 0.25,
                 mlp_ratio: int = 4,
                 pretrained: bool = False):
        super().__init__()
        assert len(stage_channels) == len(num_blocks) == len(n_heads) == len(strides)
        self.num_frames = num_frames
        self.window_size = window_size

        # Stem : /4 spatial, T inchangé
        self.stem = nn.Sequential(
            nn.Conv3d(3, stem_channels // 2, kernel_size=(3, 3, 3),
                      stride=(1, 2, 2), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(stem_channels // 2),
            nn.GELU(),
            nn.Conv3d(stem_channels // 2, stem_channels, kernel_size=(3, 3, 3),
                      stride=(1, 2, 2), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(stem_channels),
            nn.GELU(),
        )

        # Stochastic depth linéaire
        total_blocks = sum(num_blocks)
        dpr = [r.item() for r in torch.linspace(0, drop_path_rate, max(total_blocks, 1))]

        stages = []
        idx = 0
        prev_c = stem_channels
        for stage_idx, (out_c, nb, nh, st) in enumerate(zip(stage_channels, num_blocks, n_heads, strides)):
            blocks = []
            for j in range(nb):
                block_stride = st if j == 0 else 1
                blocks.append(MaxViTBlock(
                    in_c=prev_c if j == 0 else out_c,
                    out_c=out_c,
                    n_heads=nh,
                    window_size=window_size,
                    stride=block_stride,
                    dropout=dropout,
                    drop_path=dpr[idx],
                    mlp_ratio=mlp_ratio,
                    expand=expand,
                    se_ratio=se_ratio,
                ))
                idx += 1
            stages.append(nn.Sequential(*blocks))
            prev_c = out_c
        self.stages = nn.ModuleList(stages)

        self.final_norm = nn.LayerNorm(prev_c)
        self.head_dropout = nn.Dropout(head_dropout)
        self.head = nn.Linear(prev_c, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # (B, T, C, H, W) -> (B, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        # global avg pool sur (T, H, W) -> (B, C)
        x = x.mean(dim=(2, 3, 4))
        x = self.final_norm(x)
        x = self.head_dropout(x)
        return self.head(x)

    # Compat avec train.py : pas d'init de backbone externe (from scratch)
    def set_backbone_trainable(self, trainable: bool) -> None:
        for module in (self.stem, *self.stages):
            for p in module.parameters():
                p.requires_grad = trainable
