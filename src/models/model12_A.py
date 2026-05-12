import torch
import torch.nn as nn
import numpy as np


class PatchEmbeddings(nn.Module):
    def __init__(self, img_size=96, patch_size=16, hidden_dim=512):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.patch = nn.Conv2d(3, hidden_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.patch(x)
        # x = [batch size, hidden dim, H_patch, W_patch]
        x = x.flatten(2).transpose(1, 2)
        return x

class Attention(nn.Module):
    def __init__(self, x_to_dim, x_from_dim, hidden_dim):
        super(Attention, self).__init__()
        self.x_to_dim = x_to_dim
        self.x_from_dim = x_from_dim
        self.hidden_dim = hidden_dim
        self.query_matrix = nn.Linear(x_to_dim, hidden_dim)
        self.key_matrix = nn.Linear(x_from_dim, hidden_dim)
        self.value_matrix = nn.Linear(x_from_dim, hidden_dim)
        
    def forward(self, x_to, x_from):
        # x_to = [batch size, x_to_len, x_to_dim]
        # x_from = [batch size, x_from_len, x_from_dim]

        Q = self.query_matrix(x_to)
        K = self.key_matrix(x_from)
        V = self.value_matrix(x_from)
        d_k = Q.shape[-1]

        similarity_matrix = torch.einsum("bik,bjk->bij", Q, K)/ np.sqrt(d_k)
        attn_weights = torch.softmax(similarity_matrix, dim=-1)
        output = torch.einsum("bij,bjk->bik", attn_weights, V)
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, x_to_dim, x_from_dim, hidden_dim, n_heads,
                 attn_dropout=0.0, proj_dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.x_to_dim = x_to_dim
        self.x_from_dim = x_from_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.query_matrix = nn.Linear(x_to_dim, hidden_dim)
        self.key_matrix = nn.Linear(x_from_dim, hidden_dim)
        self.value_matrix = nn.Linear(x_from_dim, hidden_dim)

        self.output_projection = nn.Linear(hidden_dim, hidden_dim)

        # REGULARIZATION
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj_dropout = nn.Dropout(proj_dropout)

    def forward(self, x_to, x_from):
        batch_size, x_to_len, _ = x_to.shape
        _, x_from_len, _ = x_from.shape

        Q = self.query_matrix(x_to)
        K = self.key_matrix(x_from)
        V = self.value_matrix(x_from)

        Q = Q.view(batch_size, x_to_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, x_from_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, x_from_len, self.n_heads, self.head_dim).transpose(1, 2)

        similarity_matrix = torch.einsum("bdik,bdjk->bdij", Q, K) / self.head_dim**0.5
        attn_weights = torch.softmax(similarity_matrix, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)          # REGULARIZATION
        output = torch.einsum("bdij,bdjk->bdik", attn_weights, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, x_to_len, self.hidden_dim)
        output = self.output_projection(output)
        output = self.proj_dropout(output)                       # REGULARIZATION
        return output


class MultiHeadSelfAttention(MultiHeadAttention):
    def __init__(self, x_dim, hidden_dim, n_heads, attn_dropout=0.0, proj_dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__(x_dim, x_dim, hidden_dim, n_heads, attn_dropout, proj_dropout)

    def forward(self, x):
        # x = [batch size, x_len, x_dim]
        return super().forward(x, x)


class FFN(nn.Module):
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


class TransformerEncoderBlock(nn.Module):
    def __init__(self, data_dim, hidden_dim, n_heads, dropout_rate=0.1, attn_dropout=0.0):
        super().__init__()
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim 
        self.n_heads = n_heads 
        self.dropout_rate = dropout_rate

        self.norm1 = nn.LayerNorm(self.hidden_dim)
        self.norm2 = nn.LayerNorm(self.hidden_dim)
        self.attention = MultiHeadSelfAttention(self.hidden_dim, self.hidden_dim, self.n_heads, attn_dropout=attn_dropout, proj_dropout=dropout_rate)
        self.MLP = FFN(self.hidden_dim, dropout_rate)

        ### REGULARIZATION
        self.resid_dropout = nn.Dropout(dropout_rate)


    def forward(self, x):
        # x = [batch size, x_len, hidden dim]
        x_norm = self.norm1(x)
        attn = self.attention(x_norm)
        x1 = x + self.resid_dropout(attn)   # REGULARIZATION
        x_norm_ = self.norm2(x1)
        output = self.MLP(x_norm_)
        output = x1 + self.resid_dropout(output)  # REGULARIZATION
        return output 


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, max_len):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_len = max_len
        self.PE = nn.Embedding(max_len, self.hidden_dim)

    def forward(self, x):
        # x = [batch size, seq len, hidden dim]
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        x = x + self.PE(positions)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)
        
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = nn.Identity()

        self.relu = nn.ReLU()

    def forward(self, x):
        identity = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + identity)
        return out


class CNNSpatialEncoder2(nn.Module):
    """
    ResNet-18 style spatial encoder.
    Stem: 7x7 conv s=2 (224->112) -> MaxPool 3x3 s=2 (112->56)
    Stage1: 2 BasicBlocks, 64ch,  56->56
    Stage2: 2 BasicBlocks, 128ch, 56->28
    Stage3: 2 BasicBlocks, 256ch, 28->14
    Stage4: 2 BasicBlocks, 512ch, 14->7
    GAP -> Linear(512, hidden_dim)
    """
    def __init__(self, hidden_dim=512):
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
        self.stage4 = self._make_stage(256, 512, num_blocks=2, stride=2)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(512, hidden_dim)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_stage(self, in_channels, out_channels, num_blocks, stride):
        layers = [BasicBlock(in_channels, out_channels, stride=stride)]
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        # x: (B*T, 3, H, W), H=W=224 attendu
        x = self.stem(x)      # (B*T, 64, 56, 56)
        x = self.stage1(x)    # (B*T, 64, 56, 56)
        x = self.stage2(x)    # (B*T, 128, 28, 28)
        x = self.stage3(x)    # (B*T, 256, 14, 14)
        x = self.stage4(x)    # (B*T, 512, 7, 7)
        x = self.global_pool(x).flatten(1)  # (B*T, 512)
        x = self.proj(x)                    # (B*T, hidden_dim)
        return x


class VideoViT_CNN_2(nn.Module):
    def __init__(self, img_size=224, patch_size=16, hidden_dim=512,
                 n_heads=8, n_spatial_layers=6, n_temporal_layers=2,
                 dropout_rate=0.1, num_classes=33, num_frames=16, pretrained=False):
        super().__init__()
        n_patches = (img_size // patch_size) ** 2

        # Encodeur spatial (partagé entre toutes les frames)
        self.spatial_encoder = CNNSpatialEncoder2(hidden_dim)

        # Encodeur temporel (sur les T CLS tokens)
        self.temporal_pos_enc = LearnedPositionalEncoding(hidden_dim, num_frames)
        self.temporal_layers = nn.ModuleList([
            TransformerEncoderBlock(hidden_dim, hidden_dim, n_heads, dropout_rate)
            for _ in range(n_temporal_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x : (B, T, C, H, W)
        B, T, C, H, W = x.shape

        # --- Encodage spatial grace au CNN custom ---
        x = x.view(B * T, C, H, W)
        x = self.spatial_encoder(x)     # (B*T, hidden_dim)
        x = x.view(B, T, -1)           # (B, T, hidden_dim)

        # --- Encodage temporel ---
        x = self.temporal_pos_enc(x)
        for layer in self.temporal_layers:
            x = layer(x)
        x = self.norm(x)
        x = x.mean(dim=1)                                    # (B, D)

        return self.classifier(x)                            # (B, num_classes)