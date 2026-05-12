""" First model architechture - ViT pretrained
   - **Input:** `(B, T, C, H, W)`  
   - **BackBone : MViT_V1 
   - **Output:** logits `(B, num_classes)`.

Forward (conceptually):
    Input:  (batch, time, C, H, W)
    Reshape: (batch * time, C, H, W)  # each frame is an independent image
    Backbone: ResNet18 up to global average pool -> (batch * time, 512, 1, 1)
    Flatten: (batch * time, 512)
    Reshape: (batch, time, 512)
    Mean over time: (batch, 512)
    Linear classifier: (batch, num_classes)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class MViT_test_1(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = False, freeze_backbone: bool = False) -> None:
        super().__init__()
        weights = models.video.MViT_V1_B_Weights.KINETICS400_V1 if pretrained else None
        backbone = models.video.mvit_v1_b(
            weights=weights
            )

        # MVit a 768 dim de sortie pour le dernier block (kinetics400), pas besoin de la FC finale
        feature_dim = 768
        backbone.head = nn.Identity()  # Remove the final classification head
        self.backbone = backbone
        self.classifier = nn.Linear(feature_dim, num_classes)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            
    def forward(self, video_batch: torch.Tensor) -> torch.Tensor:
        """
        video_batch: (batch_size, T, C, H, W)
        returns logits: (batch_size, num_classes)
        """
        # MViT attend une entrée de type vidéo (B,C,T,H,W)
        video_batch = video_batch.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
        features = self.backbone(video_batch)  # (B, feature_dim)
        logits = self.classifier(features)  # (B, num_classes)
        
        return logits
