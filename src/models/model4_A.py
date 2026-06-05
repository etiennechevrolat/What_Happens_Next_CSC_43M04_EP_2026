"""
model4_A : CNN spatial + Temporal Attention (4 tokens) avec CLS.

Brigues réutilisées :
- BasicBlock (2D ResNet, model2_A)
- DropPath / Stochastic Depth (model3_A)
- TransformerEncoderBlock + MultiHeadSelfAttention + FFN (model2_A)

Différences cs model2 et model3 :
- Backbone spatial allégé : 64 -> 128 -> 256 canaux (au lieu de 512)
- Pooling spatial AVANT l'attention temporelle : on passe de 196 tokens à 4
- 1 token CLS + pos embedding de 5 positions (CLS + 4 frames)
- 1 ou 2 blocs Transformer suffisent (1 seul est souvent optimal sur 5 tokens)
- DropPath linéaire 0 -> drop_path_rate à travers les blocs Transformer
"""

import torch
import torch.nn as nn
import numpy as np


class DropPath(nn.Module):
    """Stochastic Depth (identique à model3_A)."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor + keep_prob)
        return x * random_tensor / keep_prob

    def extra_repr(self):
        return f"drop_prob={self.drop_prob:.2f}"


class BasicBlock(nn.Module):
    """ResNet 2D BasicBlock reutilisé depuis model2."""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + identity)
        return out


class FFN(nn.Module):
    """Feed-forward du Transformer (identique à model2_A)."""
    def __init__(self, n_embd, dropout=0.1, expansion_factor=4):
        super().__init__()
        self.fc1 = nn.Linear(n_embd, expansion_factor * n_embd)
        self.fc2 = nn.Linear(expansion_factor * n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class MultiHeadSelfAttention(nn.Module):
    """
    Self-attention multi-tête (version simplifiée et inline pour clarté).
    Reprend la même logique que ta MultiHeadAttention de model2_A.
    """
    def __init__(self, hidden_dim, n_heads, attn_dropout=0.0, proj_dropout=0.1):
        super().__init__()
        assert hidden_dim % n_heads == 0
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim, bias=True)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj_dropout = nn.Dropout(proj_dropout)

    def forward(self, x):
        # x : (B, N, D)
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # (B, n_heads, N, head_dim)

        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, N, D)
        out = self.proj(out)
        out = self.proj_dropout(out)
        return out


class TransformerEncoderBlock(nn.Module):
    """
    Bloc Transformer (Pre-Norm), version reprise de model2_A,
    + DropPath sur les deux branches résiduelles.
    """
    def __init__(self, hidden_dim, n_heads, dropout_rate=0.1, attn_dropout=0.0,
                 drop_path_p: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = MultiHeadSelfAttention(hidden_dim, n_heads,
                                            attn_dropout=attn_dropout,
                                            proj_dropout=dropout_rate)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = FFN(hidden_dim, dropout=dropout_rate)
        self.drop_path = DropPath(drop_path_p) if drop_path_p > 0.0 else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class SpatialEncoder(nn.Module):
    """
    - 64 -> 128 -> 256 canaux
    - 4 stages
    - Global Avg Pool spatial à la fin -> 1 vecteur par frame
    """
    def __init__(self, embed_dim=256):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.stage1 = self._make_stage(64,  64,  num_blocks=2, stride=1)
        self.stage2 = self._make_stage(64,  128, num_blocks=2, stride=2)
        self.stage3 = self._make_stage(128, 256, num_blocks=2, stride=2)

        # Projection vers embed_dim si différent de 256
        if embed_dim != 256:
            self.proj = nn.Conv2d(256, embed_dim, kernel_size=1)
        else:
            self.proj = nn.Identity()

        self.pool = nn.AdaptiveAvgPool2d(1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_stage(self, in_c, out_c, num_blocks, stride):
        layers = [BasicBlock(in_c, out_c, stride=stride)]
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_c, out_c, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        # x : (B*T, 3, 224, 224)
        x = self.stem(x)      # (B*T, 64,  56, 56)
        x = self.stage1(x)    # (B*T, 64,  56, 56)
        x = self.stage2(x)    # (B*T, 128, 28, 28)
        x = self.stage3(x)    # (B*T, 256, 14, 14)
        x = self.proj(x)      # (B*T, embed_dim, 14, 14)
        x = self.pool(x)      # (B*T, embed_dim, 1, 1)
        return x.flatten(1)   # (B*T, embed_dim)


class CompactVideoModel(nn.Module):
    """
    Pipeline complet :
      (B, T, C, H, W)
      -> [encode chaque frame avec SpatialEncoder]   (B*T, D)
      -> (B, T, D) + CLS + pos embed                          (B, T+1, D)
      -> n_temporal_layers blocs Transformer
      -> CLS token                                            (B, D)
      -> Linear -> logits                                     (B, num_classes)

    Args:
        num_classes      : nb de classes (33)
        num_frames       : T (= 4)
        embed_dim        : dim des features et du transformer (default 256)
        n_heads          : têtes d'attention (default 4)
        n_temporal_layers: nb de blocs Transformer temporels (1 ou 2 max)
        dropout_rate     : dropout dans le transformer (attention proj, ffn, head)
        drop_path_rate   : taux max de stochastic depth (linéaire 0 -> drop_path_rate)
        head_dropout     : dropout devant la fc finale
        pretrained       : ignoré (from scratch)
    """
    def __init__(self,
                 num_classes=33,
                 num_frames=4,
                 embed_dim=256,
                 n_heads=4,
                 n_temporal_layers=1,
                 dropout_rate=0.2,
                 drop_path_rate=0.1,
                 head_dropout=0.4,
                 pretrained=False):
        super().__init__()
        self.num_frames = num_frames
        self.embed_dim = embed_dim

        # 1) Backbone spatial
        self.spatial_encoder = SpatialEncoder(embed_dim=embed_dim)

        # 2) CLS token + positional embedding (T+1 positions)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_frames + 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.pos_drop = nn.Dropout(dropout_rate)

        # 3) Blocs Transformer temporels avec DropPath linéaire
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_temporal_layers)]
        self.temporal_layers = nn.ModuleList([
            TransformerEncoderBlock(
                hidden_dim=embed_dim,
                n_heads=n_heads,
                dropout_rate=dropout_rate,
                attn_dropout=0.0,
                drop_path_p=dpr[i],
            )
            for i in range(n_temporal_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # 4) Head
        self.head_dropout = nn.Dropout(head_dropout)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # x : (B, T, C, H, W)
        B, T, C, H, W = x.shape

        # Encodage spatial frame-par-frame
        x = x.view(B * T, C, H, W)
        x = self.spatial_encoder(x)            # (B*T, embed_dim)
        x = x.view(B, T, -1)                   # (B, T, embed_dim)

        # Préfixer le CLS token
        cls = self.cls_token.expand(B, -1, -1) # (B, 1, embed_dim)
        x = torch.cat([cls, x], dim=1)         # (B, T+1, embed_dim)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Attention temporelle
        for layer in self.temporal_layers:
            x = layer(x)
        x = self.norm(x)

        # Classification sur le CLS
        cls_out = x[:, 0]                       # (B, embed_dim)
        cls_out = self.head_dropout(cls_out)
        return self.head(cls_out)               # (B, num_classes)