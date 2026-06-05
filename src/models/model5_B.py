import torch
import torch.nn as nn

"""
model5_B = model5_A + 3 améliorations ciblées (T=4 frames) :

  (1) Pas d'avgpool spatial avant le transformer.
      On flatten stage4 -> N = T * H' * W' tokens spatio-temporels (= 4*7*7=196).
      Le transformer voit enfin assez de tokens pour faire un travail utile.

  (2) TSM (Temporal Shift Module) dans chaque BasicBlock2Plus1D.
      Décale 1/fold_div des canaux de +1 dans T et 1/fold_div de -1.
      Coût: 0 paramètres, ~0 FLOPs. Force le mélange temporel précoce,
      très efficace quand T est petit.

  (3) Attention pooling à la place du CLS token.
      Avec N=196 tokens, agréger par attention apprise est plus stable
      qu'un CLS qui doit "rassembler" tout via self-attention.
"""


# ─── (2) TSM ──────────────────────────────────────────────────────────────────
class TemporalShift(nn.Module):
    """
    Entrée (B, C, T, H, W). Décale fold canaux gauche (t <- t+1),
    fold canaux droite (t <- t-1), le reste reste.
    """
    def __init__(self, n_channels: int, fold_div: int = 8):
        super().__init__()
        self.fold = max(1, n_channels // fold_div)

    def forward(self, x):
        fold = self.fold
        out = torch.zeros_like(x)
        out[:, :fold, :-1] = x[:, :fold, 1:]            # shift left
        out[:, fold:2 * fold, 1:] = x[:, fold:2 * fold, :-1]  # shift right
        out[:, 2 * fold:] = x[:, 2 * fold:]
        return out

    def extra_repr(self):
        return f"fold={self.fold}"


# ─── briques reprises de model5_A (inchangées) ────────────────────────────────
class FFN(nn.Module):
    def __init__(self, n_embd, dropout=0.1, expansion_factor=4):
        super().__init__()
        self.fc1 = nn.Linear(n_embd, expansion_factor * n_embd)
        self.fc2 = nn.Linear(expansion_factor * n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.fc2(self.dropout(self.activation(self.fc1(x))))


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_dim, n_heads, attn_dropout=0.0, proj_dropout=0.1):
        super().__init__()
        assert hidden_dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim, bias=True)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj_dropout = nn.Dropout(proj_dropout)

    def forward(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = self.attn_dropout(attn.softmax(dim=-1))
        out = (attn @ v).transpose(1, 2).contiguous().view(B, N, D)
        return self.proj_dropout(self.proj(out))


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.floor(torch.rand(shape, dtype=x.dtype, device=x.device) + keep_prob)
        return x * mask / keep_prob


class TransformerEncoderBlock(nn.Module):
    def __init__(self, hidden_dim, n_heads, dropout_rate=0.1, attn_dropout=0.0, drop_path_p=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = MultiHeadSelfAttention(hidden_dim, n_heads, attn_dropout, dropout_rate)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = FFN(hidden_dim, dropout=dropout_rate)
        self.drop_path = DropPath(drop_path_p) if drop_path_p > 0.0 else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Conv2Plus1D(nn.Module):
    def __init__(self, in_c, out_c, stride_t=1, stride_s=1):
        super().__init__()
        mid_c = (3 * 3 * 3 * in_c * out_c) // (3 * 3 * in_c + 3 * out_c)
        self.conv_s = nn.Conv3d(in_c, mid_c, (1, 3, 3), (1, stride_s, stride_s), (0, 1, 1), bias=False)
        self.bn_s = nn.BatchNorm3d(mid_c)
        self.conv_t = nn.Conv3d(mid_c, out_c, (3, 1, 1), (stride_t, 1, 1), (1, 0, 0), bias=False)
        self.bn_t = nn.BatchNorm3d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn_s(self.conv_s(x)))
        return self.relu(self.bn_t(self.conv_t(x)))


# ─── BasicBlock avec TSM en tête de la branche résiduelle ─────────────────────
class BasicBlock2Plus1D_TSM(nn.Module):
    def __init__(self, in_c, out_c, stride_t=1, stride_s=1,
                 dropout3d_p=0.1, drop_path_p=0.0,
                 use_tsm=True, fold_div=8):
        super().__init__()
        self.tsm = TemporalShift(in_c, fold_div) if use_tsm else nn.Identity()
        self.conv1 = Conv2Plus1D(in_c, out_c, stride_t, stride_s)
        self.conv2 = Conv2Plus1D(out_c, out_c, 1, 1)
        if stride_t != 1 or stride_s != 1 or in_c != out_c:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_c, out_c, 1, (stride_t, stride_s, stride_s), bias=False),
                nn.BatchNorm3d(out_c),
            )
        else:
            self.downsample = nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout3d(p=dropout3d_p)
        self.drop_path = DropPath(drop_path_p) if drop_path_p > 0.0 else nn.Identity()


    def forward(self, x):
        identity = self.downsample(x)
        out = self.tsm(x)              # (2) shift temporel avant conv
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.dropout(out)
        out = self.drop_path(out)
        return self.relu(out + identity)


class SpatioTemporalLayer(nn.Module):
    def __init__(self, in_c, out_c, num_blocks, stride_t=1, stride_s=1,
                 dropout3d_p=0.1, drop_path_rates=None,
                 use_tsm=True, fold_div=8):
        super().__init__()
        if drop_path_rates is None:
            drop_path_rates = [0.0] * num_blocks
        blocks = []
        for i in range(num_blocks):
            blocks.append(BasicBlock2Plus1D_TSM(
                in_c if i == 0 else out_c, out_c,
                stride_t=stride_t if i == 0 else 1,
                stride_s=stride_s if i == 0 else 1,
                dropout3d_p=dropout3d_p,
                drop_path_p=drop_path_rates[i],
                use_tsm=use_tsm, fold_div=fold_div,
            ))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


class R2Plus1DStem(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_s = nn.Conv3d(3, 45, (1, 7, 7), (1, 2, 2), (0, 3, 3), bias=False)
        self.bn_s = nn.BatchNorm3d(45)
        self.conv_t = nn.Conv3d(45, 64, (3, 1, 1), (1, 1, 1), (1, 0, 0), bias=False)
        self.bn_t = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d((1, 3, 3), (1, 2, 2), (0, 1, 1))

    def forward(self, x):
        x = self.relu(self.bn_s(self.conv_s(x)))
        x = self.relu(self.bn_t(self.conv_t(x)))
        return self.maxpool(x)


# ─── (3) Attention pooling ────────────────────────────────────────────────────
class AttentionPool(nn.Module):
    """
    Agrège (B, N, D) -> (B, D) via une attention scalaire apprise.
    Une petite MLP -> 1 logit par token, softmax sur N, somme pondérée.
    """
    def __init__(self, dim, hidden=None):
        super().__init__()
        hidden = hidden or dim
        self.score = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        w = self.score(x).squeeze(-1)            # (B, N)
        w = w.softmax(dim=1).unsqueeze(-1)       # (B, N, 1)
        return (w * x).sum(dim=1)                # (B, D)


# ─── Modèle complet ───────────────────────────────────────────────────────────
class R2Plus1D_SpatioTemporalTransformer(nn.Module):
    """
    Pipeline (T=4 fixé) :
      stem + 4 stages (avec TSM) -> (B, 512, 4, 7, 7)
      flatten spatio-temporel    -> (B, 196, 512)
      + pos_embed (factorisé t + s)
      n_temporal_layers x Transformer
      LayerNorm -> AttentionPool -> Dropout -> Linear(512, 33)
    """
    def __init__(self,
                 num_classes=33,
                 num_frames=4,
                 tokens_per_frame=49,        # 7*7 après stage4 pour input 112x112
                 dropout3d_p=0.1,
                 drop_path_rate=0.05,
                 embed_dim=512,
                 n_heads=8,
                 n_temporal_layers=2,
                 transformer_dropout=0.1,
                 transformer_drop_path_rate=0.1,
                 head_dropout=0.3,
                 use_tsm=True,
                 tsm_fold_div=8,
                 pretrained=False):
        super().__init__()
        assert embed_dim == 512
        self.num_frames = num_frames
        self.tokens_per_frame = tokens_per_frame
        self.embed_dim = embed_dim

        # Backbone
        self.stem = R2Plus1DStem()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, 8)]
        self.stage1 = SpatioTemporalLayer(64,  64,  2, 1, 1, dropout3d_p, dpr[0:2], use_tsm, tsm_fold_div)
        self.stage2 = SpatioTemporalLayer(64,  128, 2, 1, 2, dropout3d_p, dpr[2:4], use_tsm, tsm_fold_div)
        self.stage3 = SpatioTemporalLayer(128, 256, 2, 1, 2, dropout3d_p, dpr[4:6], use_tsm, tsm_fold_div)
        self.stage4 = SpatioTemporalLayer(256, 512, 2, 1, 2, dropout3d_p, dpr[6:8], use_tsm, tsm_fold_div)

        # (1) pos_embed factorisé : pos_t (1, T, 1, D) + pos_s (1, 1, S, D)
        #    -> évite N=T*S paramètres redondants et casse moins facilement à l'init.
        self.pos_t = nn.Parameter(torch.zeros(1, num_frames, 1, embed_dim))
        self.pos_s = nn.Parameter(torch.zeros(1, 1, tokens_per_frame, embed_dim))
        nn.init.trunc_normal_(self.pos_t, std=0.02)
        nn.init.trunc_normal_(self.pos_s, std=0.02)
        self.pos_drop = nn.Dropout(transformer_dropout)

        # Transformer temporel + spatial joint
        t_dpr = [x.item() for x in torch.linspace(0, transformer_drop_path_rate, n_temporal_layers)]
        self.temporal_layers = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, n_heads,
                                    dropout_rate=transformer_dropout,
                                    attn_dropout=0.0,
                                    drop_path_p=t_dpr[i])
            for i in range(n_temporal_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # (3) Attention pooling
        self.attn_pool = AttentionPool(embed_dim)

        # Head
        self.head_dropout = nn.Dropout(head_dropout)
        self.head = nn.Linear(embed_dim, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # (B, T, C, H, W) -> (B, C, T, H, W)
        B = x.shape[0]
        x = x.permute(0, 2, 1, 3, 4).contiguous()

        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)                       # (B, 512, T, H', W')

        # (1) flatten spatio-temporel : (B, 512, T, H, W) -> (B, T, H*W, 512)
        Bx, C, T, H, W = x.shape
        S = H * W
        assert T == self.num_frames, f"T={T} ≠ num_frames={self.num_frames}"
        assert S == self.tokens_per_frame, (
            f"tokens_per_frame={self.tokens_per_frame} mais H*W={S} "
            f"(adapte tokens_per_frame ou la résolution d'entrée)"
        )
        x = x.permute(0, 2, 3, 4, 1).contiguous().view(B, T, S, C)

        # pos_embed factorisé (broadcast sur T et S)
        x = x + self.pos_t + self.pos_s          # (B, T, S, D)
        x = x.view(B, T * S, C)
        x = self.pos_drop(x)

        for layer in self.temporal_layers:
            x = layer(x)
        x = self.norm(x)

        # (3) attention pooling -> (B, D)
        pooled = self.attn_pool(x)
        pooled = self.head_dropout(pooled)
        return self.head(pooled)

    # Helpers (compat model5_A)
    def load_backbone_from_model3(self, checkpoint_path: str) -> None:
        """
        Charge stem/stage1-4 depuis un checkpoint model3_A.
        ATTENTION : model3_A n'a pas de TSM. Les TSM sont sans paramètre,
        donc le chargement reste strict=False et n'aura rien d'inattendu.
        """
        ck = torch.load(checkpoint_path, map_location='cpu')
        sd = ck['model_state_dict']
        backbone_sd = {k: v for k, v in sd.items()
                       if k.startswith(('stem.', 'stage1.', 'stage2.', 'stage3.', 'stage4.'))}
        missing, unexpected = self.load_state_dict(backbone_sd, strict=False)
        allowed = ('pos_t', 'pos_s', 'temporal_layers', 'norm.', 'attn_pool', 'head.')
        bad = [k for k in missing if not k.startswith(allowed)]
        assert not bad, f"Clés backbone manquantes : {bad}"
        assert not unexpected, f"Clés inattendues : {unexpected}"
        print(f"Backbone chargé depuis {checkpoint_path} "
              f"(val_acc enregistré={ck.get('val_accuracy', 0):.4f})")

    def set_backbone_trainable(self, trainable: bool) -> None:
        for module in (self.stem, self.stage1, self.stage2, self.stage3, self.stage4):
            for p in module.parameters():
                p.requires_grad = trainable
