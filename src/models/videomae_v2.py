"""
VideoMAE V2 — implementation from scratch (Wang et al. 2023,
"VideoMAE V2: Scaling Video Masked Autoencoders with Dual Masking").

Conçu pour pretrain self-supervisé, puis transfert des poids du Transformer
encoder vers stage 3 de model6_A (dim=768, n_heads=12, 2 blocs × 2 attentions
= 4 attn-layers compatibles).

Composants
----------
1) PatchEmbed3D    : tubelet (2,16,16) -> tokens. Pour entrée (B, T=4, 3, 224, 224)
                     -> 2 × 14 × 14 = 392 tokens de dim 768.
2) Encoder ViT     : 12 blocs Pre-Norm, dim=768, heads=12. Ne reçoit que les
                     tokens visibles (tube masking, masque encoder ratio ~0.75).
3) Decoder léger   : 4 blocs Pre-Norm, dim=384, heads=6. Reçoit (visibles
                     projetés) + mask tokens, mais SEULEMENT pour un
                     sous-ensemble de positions masquées (dual masking,
                     ratio decoder ~0.5).
4) Tête de recons. : Linear(384 -> tubelet_t*patch_h*patch_w*3) pour prédire
                     les pixels normalisés par patch (norm-pix loss).

Tube masking : on tire un masque (Hp, Wp) puis on le broadcast sur Tp.
               Garantit que le modèle ne "triche" pas en interpolant
               temporellement la même position spatiale.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


# ─── Briques transformer (mêmes interfaces que model6) ───────────────────────
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


# ─── Patch embed 3D (tubelet) ────────────────────────────────────────────────
class PatchEmbed3D(nn.Module):
    """
    Conv3d(kernel=stride=(tubelet_t, patch_h, patch_w)) — tokenisation tubelet.
    """
    def __init__(self, img_size=224, num_frames=4,
                 tubelet_t=2, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        assert num_frames % tubelet_t == 0, "num_frames doit être divisible par tubelet_t"
        assert img_size % patch_size == 0, "img_size doit être divisible par patch_size"
        self.tubelet_t = tubelet_t
        self.patch_size = patch_size
        self.Tp = num_frames // tubelet_t          # 4/2 = 2
        self.Hp = img_size // patch_size           # 224/16 = 14
        self.Wp = img_size // patch_size           # 14
        self.num_patches = self.Tp * self.Hp * self.Wp  # 2*14*14 = 392

        self.proj = nn.Conv3d(
            in_chans, embed_dim,
            kernel_size=(tubelet_t, patch_size, patch_size),
            stride=(tubelet_t, patch_size, patch_size),
            bias=True,
        )

    def forward(self, x):
        # x: (B, T, C, H, W) -> (B, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = self.proj(x)                         # (B, D, Tp, Hp, Wp)
        # (B, D, Tp, Hp, Wp) -> (B, Tp*Hp*Wp, D)
        B, D, Tp, Hp, Wp = x.shape
        x = x.reshape(B, D, Tp * Hp * Wp).transpose(1, 2)
        return x


# ─── Sinusoidal 3D positional embedding ──────────────────────────────────────
def get_sincos_pos_embed_3d(embed_dim: int, Tp: int, Hp: int, Wp: int) -> torch.Tensor:
    """
    Pos-embed sinusoïdal factorisé temps × espace, comme dans VideoMAE.
    Retourne (1, Tp*Hp*Wp, embed_dim).
    """
    assert embed_dim % 4 == 0, "embed_dim doit être multiple de 4"
    dim_t = embed_dim // 4
    dim_s = embed_dim // 4  # par axe spatial, donc 2*dim_s = embed_dim/2 pour HW

    def _sincos_1d(d, pos):
        # pos: (N,), d: dim
        omega = torch.arange(d // 2, dtype=torch.float32) / (d // 2)
        omega = 1.0 / (10000 ** omega)
        out = pos[:, None] * omega[None, :]
        return torch.cat([torch.sin(out), torch.cos(out)], dim=1)  # (N, d)

    t_pos = torch.arange(Tp, dtype=torch.float32)
    h_pos = torch.arange(Hp, dtype=torch.float32)
    w_pos = torch.arange(Wp, dtype=torch.float32)

    emb_t = _sincos_1d(embed_dim // 2, t_pos)                  # (Tp, D/2)
    emb_h = _sincos_1d(embed_dim // 4, h_pos)                  # (Hp, D/4)
    emb_w = _sincos_1d(embed_dim // 4, w_pos)                  # (Wp, D/4)

    # Tile sur la grille (Tp, Hp, Wp)
    emb_t = emb_t[:, None, None, :].expand(Tp, Hp, Wp, embed_dim // 2)
    emb_h = emb_h[None, :, None, :].expand(Tp, Hp, Wp, embed_dim // 4)
    emb_w = emb_w[None, None, :, :].expand(Tp, Hp, Wp, embed_dim // 4)

    pos = torch.cat([emb_t, emb_h, emb_w], dim=-1)  # (Tp, Hp, Wp, D)
    return pos.reshape(1, Tp * Hp * Wp, embed_dim)


# ─── Tube masking + dual masking utilities ───────────────────────────────────
def tube_mask(B: int, Tp: int, Hp: int, Wp: int, mask_ratio: float, device) -> torch.Tensor:
    """
    Tube masking : même masque spatial pour toutes les frames du tubelet.
    Retourne mask (B, Tp*Hp*Wp) bool, True = MASQUÉ (caché à l'encoder).
    """
    n_spatial = Hp * Wp
    n_mask = int(round(mask_ratio * n_spatial))
    # Pour chaque batch, tire n_mask positions spatiales
    rand = torch.rand(B, n_spatial, device=device)
    idx_sort = rand.argsort(dim=1)
    spatial_mask = torch.zeros(B, n_spatial, dtype=torch.bool, device=device)
    spatial_mask.scatter_(1, idx_sort[:, :n_mask], True)
    # Broadcast sur Tp -> (B, Tp, Hp*Wp) puis (B, Tp*Hp*Wp)
    full = spatial_mask[:, None, :].expand(B, Tp, n_spatial).reshape(B, Tp * n_spatial)
    return full


def decoder_subset_mask(encoder_mask: torch.Tensor, decoder_ratio: float) -> torch.Tensor:
    """
    Dual masking: parmi les positions masquées (vues par le décodeur),
    on n'en passe qu'une fraction. Retourne (B, N) bool: True = position
    qui sera effectivement reconstruite par le décodeur.
    """
    B, N = encoder_mask.shape
    keep_per_batch = []
    for b in range(B):
        masked_idx = encoder_mask[b].nonzero(as_tuple=False).squeeze(1)
        n_keep = max(1, int(round(decoder_ratio * masked_idx.numel())))
        perm = masked_idx[torch.randperm(masked_idx.numel(), device=encoder_mask.device)]
        sel = perm[:n_keep]
        m = torch.zeros(N, dtype=torch.bool, device=encoder_mask.device)
        m[sel] = True
        keep_per_batch.append(m)
    return torch.stack(keep_per_batch, dim=0)


# ─── Modèle complet ──────────────────────────────────────────────────────────
class VideoMAEv2(nn.Module):
    """
    Args :
      img_size, num_frames, tubelet_t, patch_size, in_chans
      encoder_dim, encoder_depth, encoder_heads
      decoder_dim, decoder_depth, decoder_heads
      mask_ratio          : ratio encoder (paper: 0.75-0.90)
      decoder_mask_ratio  : ratio dual masking decoder (paper: 0.5)
      norm_pix_loss       : normalise les patches cibles par leur moyenne/std
                            (recommandé, stabilise le pretrain)
    """
    def __init__(self,
                 img_size: int = 224,
                 num_frames: int = 4,
                 tubelet_t: int = 2,
                 patch_size: int = 16,
                 in_chans: int = 3,
                 encoder_dim: int = 768,
                 encoder_depth: int = 12,
                 encoder_heads: int = 12,
                 decoder_dim: int = 384,
                 decoder_depth: int = 4,
                 decoder_heads: int = 6,
                 mlp_ratio: int = 4,
                 mask_ratio: float = 0.75,
                 decoder_mask_ratio: float = 0.5,
                 dropout: float = 0.0,
                 drop_path_rate: float = 0.0,
                 norm_pix_loss: bool = True):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.decoder_mask_ratio = decoder_mask_ratio
        self.norm_pix_loss = norm_pix_loss
        self.tubelet_t = tubelet_t
        self.patch_size = patch_size
        self.in_chans = in_chans

        # 1) Patch embed
        self.patch_embed = PatchEmbed3D(
            img_size=img_size, num_frames=num_frames,
            tubelet_t=tubelet_t, patch_size=patch_size,
            in_chans=in_chans, embed_dim=encoder_dim,
        )
        self.Tp = self.patch_embed.Tp
        self.Hp = self.patch_embed.Hp
        self.Wp = self.patch_embed.Wp
        N = self.patch_embed.num_patches

        # Pos embed sinusoïdal (fixe, non-trainable) pour encoder et decoder
        enc_pos = get_sincos_pos_embed_3d(encoder_dim, self.Tp, self.Hp, self.Wp)
        dec_pos = get_sincos_pos_embed_3d(decoder_dim, self.Tp, self.Hp, self.Wp)
        self.register_buffer("encoder_pos_embed", enc_pos, persistent=False)
        self.register_buffer("decoder_pos_embed", dec_pos, persistent=False)

        # 2) Encoder
        enc_dpr = [r.item() for r in torch.linspace(0, drop_path_rate, encoder_depth)]
        self.encoder_blocks = nn.ModuleList([
            TransformerBlock(encoder_dim, encoder_heads, mlp_ratio,
                             dropout=dropout, drop_path=enc_dpr[i])
            for i in range(encoder_depth)
        ])
        self.encoder_norm = nn.LayerNorm(encoder_dim)

        # 3) Decoder
        self.encoder_to_decoder = nn.Linear(encoder_dim, decoder_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        dec_dpr = [r.item() for r in torch.linspace(0, drop_path_rate, decoder_depth)]
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(decoder_dim, decoder_heads, mlp_ratio,
                             dropout=dropout, drop_path=dec_dpr[i])
            for i in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_dim)

        # 4) Tête de reconstruction : prédit les pixels d'un tubelet patch
        self.patch_pix_dim = tubelet_t * patch_size * patch_size * in_chans
        self.decoder_pred = nn.Linear(decoder_dim, self.patch_pix_dim, bias=True)

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
                # Init type ViT (Conv3d du patch_embed) : trunc_normal
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ── Cible pixels (par patch tubelet) ─────────────────────────────────────
    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        (B, T, C, H, W) -> (B, N, patch_pix_dim)
        avec N = Tp*Hp*Wp, ordre cohérent avec PatchEmbed3D.
        """
        B, T, C, H, W = x.shape
        Tp, Hp, Wp = self.Tp, self.Hp, self.Wp
        tt, p = self.tubelet_t, self.patch_size
        # Permute -> (B, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.reshape(B, C, Tp, tt, Hp, p, Wp, p)
        # On veut (B, Tp, Hp, Wp, tt, p, p, C) puis flatten
        x = x.permute(0, 2, 4, 6, 3, 5, 7, 1).contiguous()
        x = x.reshape(B, Tp * Hp * Wp, tt * p * p * C)
        return x

    # ── Forward pretrain (avec masking) ──────────────────────────────────────
    def forward(self, x: torch.Tensor,
                encoder_mask: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Pretrain pass. Renvoie (loss, pred, decoder_mask) :
          pred : (B, N_dec, patch_pix_dim), prédictions des positions
                 effectivement passées dans le décodeur.
          decoder_mask : (B, N) bool, positions reconstruites (True).
        Si encoder_mask n'est pas fourni, on tire un tube_mask avec self.mask_ratio.
        """
        B = x.shape[0]
        device = x.device

        # 1) tokenisation
        tokens = self.patch_embed(x)                         # (B, N, De)
        N = tokens.shape[1]
        tokens = tokens + self.encoder_pos_embed.to(tokens.dtype)

        # 2) tube masking encoder
        if encoder_mask is None:
            encoder_mask = tube_mask(B, self.Tp, self.Hp, self.Wp,
                                     self.mask_ratio, device)
        visible_mask = ~encoder_mask                          # True = visible
        # Sélection des tokens visibles. Suppose même nombre de visibles par batch
        # (vrai par construction du tube_mask).
        n_vis = int(visible_mask[0].sum().item())
        # gather des indices visibles
        vis_idx = visible_mask.nonzero(as_tuple=False).reshape(B, n_vis, 2)[..., 1]
        idx_exp = vis_idx.unsqueeze(-1).expand(-1, -1, tokens.shape[-1])
        vis_tokens = torch.gather(tokens, 1, idx_exp)         # (B, n_vis, De)

        # 3) Encoder
        for blk in self.encoder_blocks:
            vis_tokens = blk(vis_tokens)
        vis_tokens = self.encoder_norm(vis_tokens)

        # 4) Projection enc->dec
        vis_d = self.encoder_to_decoder(vis_tokens)           # (B, n_vis, Dd)

        # 5) Dual masking : sélection d'un sous-ensemble de positions masquées
        decoder_mask = decoder_subset_mask(encoder_mask, self.decoder_mask_ratio)
        n_dec_masked = int(decoder_mask[0].sum().item())

        # Construire la séquence du décodeur = visibles + (mask_tokens à n_dec_masked pos)
        # Tous reçoivent le pos_embed du décodeur correspondant à leur position originale.
        Dd = vis_d.shape[-1]
        # Pos-embed pour les positions visibles
        dec_pos = self.decoder_pos_embed.to(vis_d.dtype)      # (1, N, Dd)
        vis_pos = torch.gather(dec_pos.expand(B, -1, -1), 1, vis_idx.unsqueeze(-1).expand(-1, -1, Dd))
        vis_d = vis_d + vis_pos

        # mask tokens + pos embed pour les positions effectivement décodées
        dec_idx = decoder_mask.nonzero(as_tuple=False).reshape(B, n_dec_masked, 2)[..., 1]
        dec_idx_exp = dec_idx.unsqueeze(-1).expand(-1, -1, Dd)
        mask_tok = self.mask_token.expand(B, n_dec_masked, Dd)
        mask_pos = torch.gather(dec_pos.expand(B, -1, -1), 1, dec_idx_exp)
        mask_tok = mask_tok + mask_pos

        dec_seq = torch.cat([vis_d, mask_tok], dim=1)         # (B, n_vis + n_dec_masked, Dd)

        # 6) Decoder
        for blk in self.decoder_blocks:
            dec_seq = blk(dec_seq)
        dec_seq = self.decoder_norm(dec_seq)

        # Ne prédit QUE pour les mask tokens (les derniers n_dec_masked positions)
        pred = self.decoder_pred(dec_seq[:, n_vis:])           # (B, n_dec_masked, patch_pix_dim)

        # 7) Cible et loss (sur les positions reconstruites par le décodeur)
        target_all = self._patchify(x)                         # (B, N, patch_pix_dim)
        if self.norm_pix_loss:
            mean = target_all.mean(dim=-1, keepdim=True)
            var = target_all.var(dim=-1, keepdim=True, unbiased=False)
            target_all = (target_all - mean) / torch.sqrt(var + 1e-6)
        # Nouvel expand avec la dim pixel (≠ Dd) pour le gather sur target_all
        pix_dim = target_all.shape[-1]
        dec_idx_pix = dec_idx.unsqueeze(-1).expand(-1, -1, pix_dim)
        target = torch.gather(target_all, 1, dec_idx_pix)

        loss = ((pred - target) ** 2).mean()

        return loss, pred, decoder_mask

    # ── Forward "features" (sans masking) pour debug / probe linéaire ────────
    @torch.no_grad()
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embed(x) + self.encoder_pos_embed.to(x.dtype)
        for blk in self.encoder_blocks:
            tokens = blk(tokens)
        return self.encoder_norm(tokens)

    # ── Transfert des poids encoder vers model6 stage 3 ──────────────────────
    @torch.no_grad()
    def transfer_to_model6_stage3(self, model6: nn.Module,
                                  num_blocks_to_transfer: int = 4) -> None:
        """
        Copie les poids attention/FFN des `num_blocks_to_transfer` derniers
        blocs de l'encoder VideoMAE vers les PartitionAttention de stage 3
        de model6 (qui a 2 MaxViTBlocks × 2 attn = 4 attn layers à dim=768).

        Ordre de transfert (du dernier MAE block vers le premier model6 attn,
        ou l'inverse — au choix). On prend les derniers blocs MAE, car ils
        encodent les features les plus haut-niveau.
        """
        # model6 stage3 : model6.stages[2] = Sequential[MaxViTBlock, MaxViTBlock]
        stage3 = model6.stages[2]
        partition_attns = []
        for blk in stage3:
            partition_attns.append(blk.block_attn)
            partition_attns.append(blk.grid_attn)
        assert len(partition_attns) == 4, \
            f"stage3 attendu : 4 PartitionAttention, trouvé {len(partition_attns)}"

        # On prend les derniers `num_blocks_to_transfer` blocs MAE (les plus profonds)
        src_blocks = list(self.encoder_blocks)[-num_blocks_to_transfer:]
        assert len(src_blocks) == len(partition_attns)

        for src, dst in zip(src_blocks, partition_attns):
            # MultiHeadSelfAttention
            dst.attn.qkv.weight.copy_(src.attn.qkv.weight)
            dst.attn.qkv.bias.copy_(src.attn.qkv.bias)
            dst.attn.proj.weight.copy_(src.attn.proj.weight)
            dst.attn.proj.bias.copy_(src.attn.proj.bias)
            # LayerNorms
            dst.norm1.weight.copy_(src.norm1.weight); dst.norm1.bias.copy_(src.norm1.bias)
            dst.norm2.weight.copy_(src.norm2.weight); dst.norm2.bias.copy_(src.norm2.bias)
            # FFN
            dst.mlp.fc1.weight.copy_(src.mlp.fc1.weight); dst.mlp.fc1.bias.copy_(src.mlp.fc1.bias)
            dst.mlp.fc2.weight.copy_(src.mlp.fc2.weight); dst.mlp.fc2.bias.copy_(src.mlp.fc2.bias)

        print(f"Transféré {len(src_blocks)} blocs Transformer VideoMAEv2 "
              f"-> stage3 de model6 (4 PartitionAttention).")
