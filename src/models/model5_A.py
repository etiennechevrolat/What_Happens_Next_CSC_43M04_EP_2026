import torch
import torch.nn as nn
import numpy as np



"""
ENTRÉE : (B,t,C,H,W) puis permutation vers (B,C,t,H,W) pour les conv3d
- Stem : 1 conv spatiale (1,7,7) + 1 conv temporelle (3,1,1) + maxpool
- 4 stages de 2 blocs 2D+1D chacun, avec downsample au début de chaque stage
Nouvelle pipeline : 
stage4 → (B, 512, T, H', W')
spatial avgpool → (B, 512, T)
permute → (B, T, 512)
+ CLS + pos_embed(T+1, 512)
1-2 blocs TransformerEncoderBlock (déjà codés dans model4_A.py !)
LayerNorm → CLS token → Dropout → fc(512 → 33)

"""

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

# ── Stochastic Depth (DropPath) ───────────────────────────────────────────────
class DropPath(nn.Module):
    """
    Tue un bloc entier avec probabilité `drop_prob` pendant l'entraînement.
    Remplace le bloc par l'identity (skip connection reste active).
    Très efficace sur les ResNets — cf. "Deep Networks with Stochastic Depth".
    """
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
 
    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        # shape (B, 1, 1, 1, 1) pour broadcaster sur (B, C, T, H, W)
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor + keep_prob)
        return x * random_tensor / keep_prob
 
    def extra_repr(self):
        return f"drop_prob={self.drop_prob:.2f}"


class Conv2Plus1D(nn.Module):
    """
    Décomposition (2+1)D d'une conv 3D :
      - conv_s : conv spatiale (1, 3, 3)  -- agit frame par frame
      - conv_t : conv temporelle (3, 1, 1) -- mélange les frames voisines
    mid_c est choisi pour égaliser le budget de paramètres avec une conv 3D
    pleine (3,3,3) de in_c -> out_c.
    """
    def __init__(self, in_c, out_c, stride_t=1, stride_s=1):
        super().__init__()
        mid_c = (3 * 3 * 3 * in_c * out_c) // (3 * 3 * in_c + 3 * out_c)

        self.conv_s = nn.Conv3d(in_c, mid_c, kernel_size=(1, 3, 3),
                                stride=(1, stride_s, stride_s),
                                padding=(0, 1, 1), bias=False)
        self.bn_s = nn.BatchNorm3d(mid_c)

        self.conv_t = nn.Conv3d(mid_c, out_c, kernel_size=(3, 1, 1),
                                stride=(stride_t, 1, 1),
                                padding=(1, 0, 0), bias=False)
        self.bn_t = nn.BatchNorm3d(out_c)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn_s(self.conv_s(x)))
        x = self.relu(self.bn_t(self.conv_t(x)))
        return x

class BasicBlock2Plus1D(nn.Module):
    def __init__(self, in_c, out_c, stride_t=1, stride_s=1,
    dropout3d_p : float = 0.3, 
    drop_path_p : float = 0.0):
        super().__init__()
        self.conv1 = Conv2Plus1D(in_c, out_c, stride_t, stride_s)
        # conv2 sans stride pour préserver les dims
        self.conv2 = Conv2Plus1D(out_c, out_c, stride_t=1, stride_s=1)
        # bn2 + relu finale après skip
        if stride_t != 1 or stride_s != 1 or in_c != out_c:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_c, out_c, kernel_size=1,
                          stride=(stride_t, stride_s, stride_s), bias=False),
                nn.BatchNorm3d(out_c),
            )
        else:
            self.downsample = nn.Identity()
        self.relu = nn.ReLU(inplace=True)

        #Dropout sur la branche résiduelle
        self.dropout= nn.Dropout3d(p=dropout3d_p)

        #DropPath (Stochastic Depth)
        self.drop_path=DropPath(drop_path_p) if drop_path_p >0.0 else nn.Identity()


    def forward(self, x):
        identity = self.downsample(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.dropout(out)
        out=self.drop_path(out)

        return self.relu(out + identity)
    
class SpatioTemporalLayer(nn.Module):
    """
    Un stage du réseau, composé de plusieurs blocs BasicBlock2Plus1D.
    """
    def __init__(self, in_c, out_c, num_blocks, stride_t=1, stride_s=1,
                 dropout3d_p: float = 0.3,
                 drop_path_rates=None):
        super().__init__()
        if drop_path_rates is None:
            drop_path_rates = [0.0] * num_blocks
 
        blocks = []
        for i in range(num_blocks):
            blocks.append(
                BasicBlock2Plus1D(
                    in_c if i == 0 else out_c,
                    out_c,
                    stride_t=stride_t if i == 0 else 1,
                    stride_s=stride_s if i == 0 else 1,
                    dropout3d_p=dropout3d_p,
                    drop_path_p=drop_path_rates[i],
                )
            )
        self.blocks = nn.Sequential(*blocks)
 
    def forward(self, x):
        return self.blocks(x)


class R2Plus1DStem(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_s = nn.Conv3d(3, 45, kernel_size=(1, 7, 7),
                                stride=(1, 2, 2), padding=(0, 3, 3), bias=False)
        self.bn_s = nn.BatchNorm3d(45)
        self.conv_t = nn.Conv3d(45, 64, kernel_size=(3, 1, 1),
                                stride=(1, 1, 1), padding=(1, 0, 0), bias=False)
        self.bn_t = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2),
                                    padding=(0, 1, 1)) 

    def forward(self, x):
        x = self.relu(self.bn_s(self.conv_s(x)))
        x = self.relu(self.bn_t(self.conv_t(x)))
        x = self.maxpool(x)
        return x
    
class R2Plus1D_TransformerHead(nn.Module):
    """
    R(2+1)D-18 backbone + tête Transformer temporel.
    Entrée : (B, T, C, H, W). T tokens issus du backbone sont passés
    dans un petit transformer avec CLS, puis classification sur le CLS.

    Args:
        num_classes        : 33
        num_frames         : T (= 4)
        dropout3d_p        : dropout3d dans chaque bloc résiduel (backbone)
        drop_path_rate     : stochastic depth max dans le backbone
        embed_dim          : 512 (= sortie backbone, pas de projection)
        n_heads            : têtes du transformer
        n_temporal_layers  : nb blocs transformer (1 ou 2 max, T+1=5 tokens)
        transformer_dropout: dropout dans attn/ffn du transformer
        transformer_drop_path_rate: stochastic depth dans le transformer
        head_dropout       : dropout devant la fc finale
        pretrained         : ignoré (from scratch)
    """
    def __init__(self,
                 num_classes=33,
                 num_frames=4,
                 dropout3d_p=0.3,
                 drop_path_rate=0.1,
                 embed_dim=512,
                 n_heads=8,
                 n_temporal_layers=1,
                 transformer_dropout=0.3,
                 transformer_drop_path_rate=0.1,
                 head_dropout=0.5,
                 pretrained=False):
        super().__init__()

        assert embed_dim == 512, "embed_dim doit valoir 512 (= sortie stage4)"
        self.num_frames = num_frames
        self.embed_dim = embed_dim

        #Backbone R(2+1)D
        self.stem = R2Plus1DStem()
        num_blocks_total = 8
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_blocks_total)]

        self.stage1 = SpatioTemporalLayer(64,  64,  num_blocks=2, stride_t=1, stride_s=1,
                                          dropout3d_p=dropout3d_p, drop_path_rates=dpr[0:2])
        self.stage2 = SpatioTemporalLayer(64,  128, num_blocks=2, stride_t=1, stride_s=2,
                                          dropout3d_p=dropout3d_p, drop_path_rates=dpr[2:4])
        self.stage3 = SpatioTemporalLayer(128, 256, num_blocks=2, stride_t=1, stride_s=2,
                                          dropout3d_p=dropout3d_p, drop_path_rates=dpr[4:6])
        self.stage4 = SpatioTemporalLayer(256, 512, num_blocks=2, stride_t=1, stride_s=2,
                                          dropout3d_p=dropout3d_p, drop_path_rates=dpr[6:8])

        # Pool spatial uniquement -> garde T tokens
        self.spatial_pool = nn.AdaptiveAvgPool3d((num_frames, 1, 1))

        #Tête Transformer temporel
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_frames + 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.pos_drop = nn.Dropout(transformer_dropout)

        t_dpr = [x.item() for x in torch.linspace(0, transformer_drop_path_rate, n_temporal_layers)]
        self.temporal_layers = nn.ModuleList([
            TransformerEncoderBlock(
                hidden_dim=embed_dim,
                n_heads=n_heads,
                dropout_rate=transformer_dropout,
                attn_dropout=0.0,
                drop_path_p=t_dpr[i],
            )
            for i in range(n_temporal_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Head
        self.head_dropout = nn.Dropout(head_dropout)
        self.head = nn.Linear(embed_dim, num_classes)

        # Init Kaiming sur conv3d/bn3d du backbone
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x : (B, T, C, H, W) -> (B, C, T, H, W)
        B, T = x.shape[0], x.shape[1]
        x = x.permute(0, 2, 1, 3, 4).contiguous()

        # Backbone
        x = self.stem(x)            # (B, 64,  T, 56, 56)
        x = self.stage1(x)          # (B, 64,  T, 56, 56)
        x = self.stage2(x)          # (B, 128, T, 28, 28)
        x = self.stage3(x)          # (B, 256, T, 14, 14)
        x = self.stage4(x)          # (B, 512, T,  7,  7)


        # Pas de pool spatial : simplement flatten les 196 tokens temporels 
        x= x.flatten(2)..transpose(1,2) # (B,T*H*W, 512)

        """
        # Pool spatial -> tokens temporels
        x = self.spatial_pool(x)    # (B, 512, T, 1, 1)
        x = x.squeeze(-1).squeeze(-1)  # (B, 512, T)
        x = x.permute(0, 2, 1).contiguous()  # (B, T, 512)
        """
        
        # Prepend CLS + pos embed
        cls = self.cls_token.expand(B, -1, -1)      # (B, 1, 512)
        x = torch.cat([cls, x], dim=1)              # (B, T+1, 512)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Transformer temporel
        for layer in self.temporal_layers:
            x = layer(x)
        x = self.norm(x)

        # Classification sur CLS
        cls_out = x[:, 0]                    # (B, 512)
        cls_out = self.head_dropout(cls_out)
        return self.head(cls_out)            # (B, num_classes)

    # Helpers d'init et de freeze 
    def load_backbone_from_model3(self, checkpoint_path: str) -> None:
        """
        Charge les poids stem/stage1-4 depuis un checkpoint model3_A.
        Ignore l'ancien fc/dropout (clés 'unexpected') et laisse les nouvelles
        couches (cls_token, pos_embed, temporal_layers, norm, head) à leur init.
        """
        ck = torch.load(checkpoint_path, map_location='cpu')
        sd = ck['model_state_dict']
        # Garde uniquement les clés du backbone
        backbone_sd = {k: v for k, v in sd.items()
                       if k.startswith(('stem.', 'stage1.', 'stage2.', 'stage3.', 'stage4.'))}
        missing, unexpected = self.load_state_dict(backbone_sd, strict=False)
        # Vérifs douces : tout ce qui manque doit être des nouvelles couches
        allowed_missing_prefixes = ('cls_token', 'pos_embed', 'temporal_layers',
                                    'norm.', 'head.', 'spatial_pool')
        bad = [k for k in missing if not k.startswith(allowed_missing_prefixes)]
        assert not bad, f"Clés backbone manquantes : {bad}"
        assert not unexpected, f"Clés inattendues : {unexpected}"
        print(f"Backbone chargé depuis {checkpoint_path} "
              f"({len(backbone_sd)} tenseurs, ancien val_acc={ck.get('val_accuracy', 0):.4f})")

    def set_backbone_trainable(self, trainable: bool) -> None:
        """Phase 1 : trainable=False pour freeze. Phase 2 : True pour fine-tune."""
        for module in (self.stem, self.stage1, self.stage2, self.stage3, self.stage4):
            for p in module.parameters():
                p.requires_grad = trainable
