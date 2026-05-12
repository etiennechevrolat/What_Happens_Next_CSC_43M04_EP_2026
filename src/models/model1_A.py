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
    def __init__(self, x_to_dim, x_from_dim, hidden_dim, n_heads):
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
        
    def forward(self, x_to, x_from):
        # x_to = [batch size, x_to_len, x_to_dim]
        # x_from = [batch size, x_from_len, x_from_dim]
        
        batch_size, x_to_len, _ = x_to.shape
        batch_size, x_from_len, _ = x_from.shape

        Q = self.query_matrix(x_to)
        K = self.key_matrix(x_from)
        V = self.value_matrix(x_from)

        ### Projections sur des sous espaces de taille head_dim 
        Q = Q.view(batch_size, x_to_len, self.n_heads, self.head_dim).transpose(1,2)
        K = K.view(batch_size, x_from_len, self.n_heads, self.head_dim).transpose(1,2)
        V = V.view(batch_size, x_from_len, self.n_heads, self.head_dim).transpose(1,2)

        similarity_matrix = torch.einsum("bdik,bdjk->bdij", Q, K)/ self.head_dim**0.5
        attn_weights = torch.softmax(similarity_matrix, dim=-1)
        output = torch.einsum("bdij,bdjk->bdik", attn_weights, V)
        output = output.transpose(1,2).contiguous().view(batch_size, x_to_len, self.hidden_dim)
        output = self.output_projection(output)  
        return output


class MultiHeadSelfAttention(MultiHeadAttention):
    def __init__(self, x_dim, hidden_dim, n_heads):
        super(MultiHeadSelfAttention, self).__init__(x_dim, x_dim, hidden_dim, n_heads)

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
    def __init__(self, data_dim, hidden_dim, n_heads, dropout_rate=0.1):
        super().__init__()
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim 
        self.n_heads = n_heads 
        self.dropout_rate = dropout_rate

        self.norm1 = nn.LayerNorm(self.hidden_dim)
        self.norm2 = nn.LayerNorm(self.hidden_dim)
        self.attention = MultiHeadSelfAttention(self.hidden_dim, self.hidden_dim, self.n_heads)
        self.MLP = FFN(self.hidden_dim, dropout_rate)


    def forward(self, x):
        # x = [batch size, x_len, hidden dim]
        x_norm = self.norm1(x)
        attn = self.attention(x_norm)
        x1 = x + attn 
        x_norm_ = self.norm2(x1)
        output = self.MLP(x_norm_)
        output = output + x1
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




### Video encoder using cnn to extract spatial features, then transformer to extract temporal features
class CNNSpatialEncoder(nn.Module):
    # (B*T, 3, H, W) -> (B*T, hidden_dim)
    ## Par défaut B = 8, T = 4 , hidden_dim = 512, H = W = 224
    ### Donc en entrée (32, 3, 224, 224)
    def __init__(self, hidden_dim=512):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2) 
        self.pool1 = nn.MaxPool2d(kernel_size=5, stride=2, padding=2)  
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=5, stride=2, padding=2) # 16x16
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=192, kernel_size=5, padding=2)
        self.pool3 = nn.MaxPool2d(kernel_size=5, stride=2, padding=2) # 8x8
        
        self.skip1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1)
        self.skip2 = nn.Conv2d(in_channels=64, out_channels=192, kernel_size=1)
    
        self.dropout_conv = nn.Dropout2d(0.2)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(192, hidden_dim)
    
    def forward(self, x):
        #x = [batch size, 3, H, W]
        out1 = self.pool1(self.dropout_conv(torch.relu(self.conv1(x))))

        out2 = torch.relu(self.conv2(out1))
        out2 = self.dropout_conv(out2)
        out2 = self.pool2(out2 + self.skip1(out1))

        out3 = torch.relu(self.conv3(out2))
        out3 = self.dropout_conv(out3)
        out3 = self.pool3(out3 + self.skip2(out2))
    

        x = self.global_pool(out3).flatten(1) # [batch size, 192]
        x = self.proj(x) # [batch size, hidden dim]
        return x

class VideoViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, hidden_dim=512,
                 n_heads=8, n_spatial_layers=6, n_temporal_layers=2,
                 dropout_rate=0.1, num_classes=33, num_frames=16, pretrained=False):
        super().__init__()
        n_patches = (img_size // patch_size) ** 2

        # Encodeur spatial (partagé entre toutes les frames)
        self.spatial_encoder = CNNSpatialEncoder(hidden_dim)

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