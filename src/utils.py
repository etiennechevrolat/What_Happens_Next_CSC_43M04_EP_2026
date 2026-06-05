"""
Small helpers: reproducibility, image transforms, and metric computation.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image, ImageFilter


def set_seed(seed: int) -> None:
    """Make runs reproducible (as far as CUDA allows)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_transforms(
    image_size: int = 224,
    is_training: bool = True,
    use_imagenet_norm: bool = True,
) -> transforms.Compose:
    """
    Standard torchvision pipeline for single RGB frames.

    use_imagenet_norm:
        True  -> mean/std from ImageNet (usual when pretrained=True)
        False -> still scale to [0,1]; you can swap norms if you prefer
    """
    if use_imagenet_norm:
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
    else:
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    if is_training:
        return transforms.Compose(
            [
                transforms.Resize((int(image_size * 1.1), int(image_size * 1.1))),
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomGrayscale(p=0.05),
                transforms.ToTensor(),
                normalize,
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize((int(image_size * 1.1), int(image_size * 1.1))),
            transforms.ToTensor(),
            normalize,
        ]
    )


@torch.no_grad()
def accuracy_topk(
    logits: torch.Tensor,
    targets: torch.Tensor,
    topk: Tuple[int, ...] = (1, 5),
) -> Tuple[torch.Tensor, ...]:
    """
    Compute top-k correctness for each k in topk.

    logits: (batch_size, num_classes)
    targets: (batch_size,) integer class indices
    Returns a tuple of tensors, each shape (1,) with accuracy in [0, 1].
    """
    max_k = max(topk)
    batch_size = targets.size(0)

    # (batch_size, max_k) indices of top predictions
    _, predictions = logits.topk(max_k, dim=1, largest=True, sorted=True)
    predictions = predictions.t()  # (max_k, batch_size)
    correct = predictions.eq(targets.view(1, -1).expand_as(predictions))

    accuracies = []
    for k in topk:
        # Any hit in the top-k row slice counts
        accuracies.append(correct[:k].reshape(-1).float().sum() / batch_size)
    return tuple(accuracies)


def split_train_val(
    samples: List[Tuple[Path, int]],
    val_ratio: float,
    seed: int,
) -> Tuple[List[Tuple[Path, int]], List[Tuple[Path, int]]]:
    """
    Shuffle then split a list of (video_path, label) into train and validation portions.

    Mirrors a standard random hold-out so train.py and evaluate.py stay consistent.
    """
    rng = random.Random(seed)
    shuffled = list(samples)
    rng.shuffle(shuffled)

    if val_ratio <= 0.0:
        return shuffled, []

    n_val = int(round(len(shuffled) * val_ratio))
    n_val = max(1, n_val) if len(shuffled) > 1 else 0

    val_samples = shuffled[:n_val]
    train_samples = shuffled[n_val:]
    if len(train_samples) == 0:
        train_samples = val_samples[:-1]
        val_samples = val_samples[-1:]

    return train_samples, val_samples


def frame_interpolation(frames, alpha_range=(0.3, 0.7)):
    new_frames= []
    for i in range(len(frames)-1):
        alpha = random.uniform(*alpha_range)
        blended = Image.blend(frames[i], frames[i+1], alpha)
        new_frames.append(blended)
    new_frames.append(frames[-1])
    return new_frames # 4 nouvelles frames comme blend de frames consécutives 

class VideoTransform:
    """
    Transform vidéo : prend une liste de PIL frames et renvoie un tensor (T, C, H, W).
    En entraînement, les paramètres d'augmentation aléatoires sont **partagés**
    entre toutes les frames d'un même clip (cohérence temporelle).
    En éval, transform déterministe.
    """
    def __init__(
        self,
        image_size: int = 224,
        is_training: bool = True,
        use_imagenet_norm: bool = True,
    ):
        self.image_size = image_size
        self.is_training = is_training

        if use_imagenet_norm:
            self.mean = [0.485, 0.456, 0.406]
            self.std = [0.229, 0.224, 0.225]
        else:
            self.mean = [0.5, 0.5, 0.5]
            self.std = [0.5, 0.5, 0.5]

        self.resize_size = int(image_size * 1.1)

    def __call__(self, frames: List[Image.Image]) -> torch.Tensor:
       
        frames = [F.resize(f, [self.resize_size, self.resize_size]) for f in frames]

        if self.is_training:
           
            i, j, h, w = transforms.RandomCrop.get_params(
                frames[0], output_size=(self.image_size, self.image_size)
            )
            frames = [F.crop(f, i, j, h, w) for f in frames]

        
            if random.random() < 0.5:
                frames = [F.hflip(f) for f in frames]


            if random.random() < 0.3 :
                angle = random.uniform(-10, 10)
                frames = [F.rotate(f, angle) for f in frames]

            
            if random.random() < 0.2:
                w_img, h_img = frames[0].size
                startpoints, endpoints = transforms.RandomPerspective.get_params(
                    w_img, h_img, distortion_scale=0.4
                )
                frames = [F.perspective(f, startpoints, endpoints) for f in frames]

            if random.random() < 0.2:
                radius = random.uniform(0.1, 2.0)
                frames = [f.filter(ImageFilter.GaussianBlur(radius=radius)) for f in frames]

            if random.random() < 0.05:
                threshold = random.randint(64, 192)
                frames = [F.solarize(f, threshold=threshold) for f in frames]

            if random.random() < 0.05:
                bits = random.randint(4, 7)
                frames = [F.posterize(f, bits) for f in frames]
            
            fn_idx, brightness, contrast, saturation, hue = transforms.ColorJitter.get_params(
                    brightness=[0.8, 1.2],
                    contrast=[0.8, 1.2],
                    saturation=[0.9, 1.1],
                    hue=[-0.05, 0.05],
                )

            new_frames = []
            for f in frames:
                for fn_id in fn_idx:
                    if fn_id == 0 and brightness is not None:
                        f = F.adjust_brightness(f, brightness)
                    elif fn_id == 1 and contrast is not None:
                        f = F.adjust_contrast(f, contrast)
                    elif fn_id == 2 and saturation is not None:
                        f = F.adjust_saturation(f, saturation)
                    elif fn_id == 3 and hue is not None:
                        f = F.adjust_hue(f, hue)
                new_frames.append(f)
            frames = new_frames

            if random.random() < 0.05 :
                enriched=[]
                blended_frames = frame_interpolation(frames)
                for i in range(len(frames)-1):
                    enriched.append(frames[i])
                    enriched.append(blended_frames[i])
                enriched.append(frames[-1])

                idx = sorted(random.sample(range(len(enriched)), 4))
                frames = [enriched[i] for i in idx]

            # RandomGrayscale : 1 seul tirage pour tout le clip
            if random.random() < 0.05:
                frames = [F.rgb_to_grayscale(f, num_output_channels=3) for f in frames]
        else:
            # Center crop pour avoir exactement image_size en eval
            frames = [F.center_crop(f, [self.image_size, self.image_size]) for f in frames]

        # ToTensor + Normalize, frame par frame (déterministe, pas de cohérence à préserver)
        tensors = []
        for f in frames:
            t = F.to_tensor(f)
            t = F.normalize(t, self.mean, self.std)
            tensors.append(t)
        
        video = torch.stack(tensors, dim=0) # (T, C, H, W)
        if self.is_training and random.random() < 0.5 :
            i, j, h_e, w_e, v = transforms.RandomErasing.get_params(
                    video[0], scale=(0.02, 0.2), ratio=(0.3, 3.3), value=[0]
                )

            video[:, :, i:i+h_e, j:j+w_e] = v  # broadcast sur l'axe T
        return video


def build_transforms(
    image_size: int = 224,
    is_training: bool = True,
    use_imagenet_norm: bool = True,
) -> VideoTransform:
    return VideoTransform(
        image_size=image_size,
        is_training=is_training,
        use_imagenet_norm=use_imagenet_norm,
    )