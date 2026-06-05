import torch
import torch.nn as nn
import numpy as np



"""
ENTRÉE : (B,t,C,H,W) puis permutation vers (B,C,t,H,W) pour les conv3d
- Stem : 1 conv spatiale (1,7,7) + 1 conv temporelle (3,1,1) + maxpool
- 4 stages de 2 blocs 2D+1D chacun, avec downsample au début de chaque stage
- Global avg pool + FC pour classification
"""


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
    

class R2Plus1D(nn.Module):
    """
    R(2+1)D-18 from scratch pour classification vidéo.
    Entrée : (B, 3, T, H, W) avec T=4, H=W=224 recommandé.

    dropout : dropput avant la fc
    dropout3d_p: dropout3d dans chaque bloc résiduel
    drop_path_rate : taux max de stochastic depth 

    """
    def __init__(self, num_classes=33, dropout=0.5, dropout3d_p= 0.3, drop_path_rate=0.1, pretrained=False, num_frames=4):
        super().__init__()
        self.stem = R2Plus1DStem()  # (B, 64, T, H, W)
        
        # Stochastic depth : taux linéaire de 0 -> drop_path_rate sur les 8 blocs
        num_blocks_total = 8  # 4 stages × 2 blocs
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_blocks_total)]
 
        self.stage1 = SpatioTemporalLayer(64,  64,  num_blocks=2, stride_t=1, stride_s=1,
                                          dropout3d_p=dropout3d_p,
                                          drop_path_rates=dpr[0:2])
        self.stage2 = SpatioTemporalLayer(64,  128, num_blocks=2, stride_t=1, stride_s=2,
                                          dropout3d_p=dropout3d_p,
                                          drop_path_rates=dpr[2:4])
        self.stage3 = SpatioTemporalLayer(128, 256, num_blocks=2, stride_t=1, stride_s=2,
                                          dropout3d_p=dropout3d_p,
                                          drop_path_rates=dpr[4:6])
        self.stage4 = SpatioTemporalLayer(256, 512, num_blocks=2, stride_t=1, stride_s=2,
                                          dropout3d_p=dropout3d_p,
                                          drop_path_rates=dpr[6:8])
 
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(512, num_classes)
        # Init Kaiming
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x attendu en (B, T, C, H, W) — convention de ton dataset.
        # On permute vers (B, C, T, H, W) pour les Conv3d.
        x = x.permute(0, 2, 1, 3, 4).contiguous()

        x = self.stem(x)        # (B, 64, T,   H, W)
        x = self.stage1(x)      # (B, 64, T,   H, W)
        x = self.stage2(x)      # (B, 128, T/2, H/2, W/2)
        x = self.stage3(x)      # (B, 256, T/4, H/4, W/4)
        x = self.stage4(x)      # (B, 512, T/8, H/8, W/8)

        x = self.avgpool(x)     # (B, 512, 1, 1, 1)
        x = torch.flatten(x, 1) # (B, 512)
        x = self.dropout(x)
        x = self.fc(x)          # (B, num_classes)
        return x