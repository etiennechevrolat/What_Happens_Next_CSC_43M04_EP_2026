"""
model7_A — VideoViT (finetune classifier de VideoMAEv2)

L'archi est rigoureusement identique à l'encoder VideoMAEv2 (patch_embed
3D + N blocs Transformer Pre-Norm). À cela s'ajoute :
  - une tête de classification : LayerNorm -> global avg pool sur tokens
    -> Dropout -> Linear(embed_dim, num_classes)

Ainsi `load_from_videomae_v2(checkpoint_path)` recopie 100 % du backbone
pretrain (patch_embed + 12 blocs Transformer + final LN) et n'initialise
à neuf que la fc finale. Aucun paramètre pretrain n'est jeté côté model7.

Notes sur le finetune (recos VideoMAE) :
  - lr base ~ 5e-4 × bs / 256 (linear scaling) avec layer-wise decay 0.65-0.75
  - drop_path 0.1-0.2
  - mixup + cutmix activés
  - label_smoothing 0.1
  - RandomErasing 0.25
  (Layer-wise LR decay non implémenté ici ; à ajouter si overfit en finetune.)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


# ─── DropPath / FFN / MHSA (mêmes briques que videomae_v2 pour compat) ───────
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


class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio=4, dropout=0.0, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, n_heads, 0.0, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = FFN(dim, dropout=dropout, expansion=mlp_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# ─── Patch embed 3D + pos embed sinusoïdal ───────────────────────────────────
class PatchEmbed3D(nn.Module):
    def __init__(self, img_size=224, num_frames=4,
                 tubelet_t=2, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        assert num_frames % tubelet_t == 0
        assert img_size % patch_size == 0
        self.tubelet_t = tubelet_t
        self.patch_size = patch_size
        self.Tp = num_frames // tubelet_t
        self.Hp = img_size // patch_size
        self.Wp = img_size // patch_size
        self.num_patches = self.Tp * self.Hp * self.Wp
        self.proj = nn.Conv3d(in_chans, embed_dim,
                              kernel_size=(tubelet_t, patch_size, patch_size),
                              stride=(tubelet_t, patch_size, patch_size),
                              bias=True)

    def forward(self, x):
        # (B, T, C, H, W) -> (B, C, T, H, W) -> (B, D, Tp, Hp, Wp)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = self.proj(x)
        B, D, Tp, Hp, Wp = x.shape
        return x.reshape(B, D, Tp * Hp * Wp).transpose(1, 2)


def get_sincos_pos_embed_3d(embed_dim: int, Tp: int, Hp: int, Wp: int) -> torch.Tensor:
    assert embed_dim % 4 == 0
    def _sincos_1d(d, pos):
        omega = torch.arange(d // 2, dtype=torch.float32) / (d // 2)
        omega = 1.0 / (10000 ** omega)
        out = pos[:, None] * omega[None, :]
        return torch.cat([torch.sin(out), torch.cos(out)], dim=1)

    t_pos = torch.arange(Tp, dtype=torch.float32)
    h_pos = torch.arange(Hp, dtype=torch.float32)
    w_pos = torch.arange(Wp, dtype=torch.float32)

    emb_t = _sincos_1d(embed_dim // 2, t_pos)
    emb_h = _sincos_1d(embed_dim // 4, h_pos)
    emb_w = _sincos_1d(embed_dim // 4, w_pos)

    emb_t = emb_t[:, None, None, :].expand(Tp, Hp, Wp, embed_dim // 2)
    emb_h = emb_h[None, :, None, :].expand(Tp, Hp, Wp, embed_dim // 4)
    emb_w = emb_w[None, None, :, :].expand(Tp, Hp, Wp, embed_dim // 4)
    pos = torch.cat([emb_t, emb_h, emb_w], dim=-1)
    return pos.reshape(1, Tp * Hp * Wp, embed_dim)


# ─── Modèle complet ──────────────────────────────────────────────────────────
class VideoViT(nn.Module):
    """
    ViT-Base vidéo pour classification, archi identique à l'encoder de
    VideoMAEv2 (charge 100 % du pretrain).

    Args:
        num_classes, num_frames, img_size, tubelet_t, patch_size, in_chans
        embed_dim, depth, n_heads, mlp_ratio
        dropout              : dropout sur attn/ffn
        drop_path_rate       : stochastic depth (linéaire 0 -> ce taux)
        head_dropout         : dropout devant la fc finale
        pool                 : 'avg' (recommandé VideoMAE) ou 'cls'
        pretrained           : ignoré ; usage : `load_from_videomae_v2(path)`
    """
    def __init__(self,
                 num_classes: int = 33,
                 num_frames: int = 4,
                 img_size: int = 224,
                 tubelet_t: int = 2,
                 patch_size: int = 16,
                 in_chans: int = 3,
                 embed_dim: int = 768,
                 depth: int = 12,
                 n_heads: int = 12,
                 mlp_ratio: int = 4,
                 dropout: float = 0.0,
                 drop_path_rate: float = 0.1,
                 head_dropout: float = 0.0,
                 pool: str = "avg",
                 pretrained: bool = False):
        super().__init__()
        assert pool in ("avg", "cls")
        self.pool = pool
        self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed3D(
            img_size=img_size, num_frames=num_frames,
            tubelet_t=tubelet_t, patch_size=patch_size,
            in_chans=in_chans, embed_dim=embed_dim,
        )
        self.Tp = self.patch_embed.Tp
        self.Hp = self.patch_embed.Hp
        self.Wp = self.patch_embed.Wp

        # Pos embed sinusoïdal (fixe, même implém que VideoMAEv2)
        pos = get_sincos_pos_embed_3d(embed_dim, self.Tp, self.Hp, self.Wp)
        self.register_buffer("pos_embed", pos, persistent=False)

        # CLS token (optionnel)
        if pool == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        else:
            self.cls_token = None

        # Transformer
        dpr = [r.item() for r in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, mlp_ratio, dropout=dropout, drop_path=dpr[i])
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Head
        self.head_dropout = nn.Dropout(head_dropout)
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv3d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Init head plus petite pour stabiliser le début du finetune (cf. ViT/DeiT)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        # (B, T, C, H, W) -> tokens (B, N, D)
        tokens = self.patch_embed(x) + self.pos_embed.to(x.dtype)
        if self.cls_token is not None:
            cls = self.cls_token.expand(tokens.shape[0], -1, -1)
            tokens = torch.cat([cls, tokens], dim=1)
        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)
        if self.pool == "cls":
            feat = tokens[:, 0]
        else:
            feat = tokens.mean(dim=1)            # global avg pool sur tokens
        feat = self.head_dropout(feat)
        return self.head(feat)

    # ── Chargement depuis pretrain VideoMAEv2 ────────────────────────────────
    def load_from_videomae_v2(self, checkpoint_path) -> None:
        """
        Charge patch_embed + tous les `encoder_blocks` + `encoder_norm` depuis
        un checkpoint VideoMAEv2 vers ce VideoViT. La tête de classif reste à
        son init. Erreur si les dims/depth ne matchent pas.
        """
        ck = torch.load(Path(checkpoint_path), map_location="cpu")
        sd = ck["model_state_dict"]

        # Remap des clés : MAE -> ViT
        remap = {}
        for k, v in sd.items():
            if k.startswith("patch_embed."):
                remap[k] = v
            elif k.startswith("encoder_blocks."):
                # encoder_blocks.{i}.* -> blocks.{i}.*
                remap[k.replace("encoder_blocks.", "blocks.")] = v
            elif k.startswith("encoder_norm."):
                remap[k.replace("encoder_norm.", "norm.")] = v
            # tout le reste (encoder_to_decoder, mask_token, decoder_*, decoder_pred)
            # est jeté par design.

        missing, unexpected = self.load_state_dict(remap, strict=False)
        # Tolérances : head + (cls_token si pool='cls')
        allowed_missing = ("head.", "cls_token")
        bad_missing = [k for k in missing if not k.startswith(allowed_missing)]
        assert not bad_missing, f"Clés backbone manquantes après load : {bad_missing}"
        assert not unexpected, f"Clés inattendues après load : {unexpected}"

        n_loaded = len(remap)
        print(f"VideoViT : chargé {n_loaded} tenseurs depuis {checkpoint_path} "
              f"(loss pretrain enregistrée={ck.get('loss', 'n/a')}). "
              f"Tête de classif reste à neuf.")

    # ── Compat train.py (pas vraiment utilisé : un VideoViT a un seul "tout") ──
    def set_backbone_trainable(self, trainable: bool) -> None:
        """Freeze tout sauf la tête de classif."""
        for p in self.patch_embed.parameters(): p.requires_grad = trainable
        for p in self.blocks.parameters():       p.requires_grad = trainable
        for p in self.norm.parameters():         p.requires_grad = trainable
