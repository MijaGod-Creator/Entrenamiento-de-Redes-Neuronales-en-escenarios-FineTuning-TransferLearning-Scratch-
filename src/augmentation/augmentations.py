from __future__ import annotations

import torch
from torchvision import transforms as T


class AdvancedAugmentation:
    """Torchvision-based augmentation block for facial emotion images."""

    def __init__(self, image_size: tuple[int, int]):
        self.pipeline = T.Compose(
            [
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(15),
                T.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.88, 1.12)),
                T.ColorJitter(brightness=0.15, contrast=0.15),
                T.RandomResizedCrop(image_size, scale=(0.78, 1.0), ratio=(0.9, 1.1)),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                T.RandomErasing(p=0.3, scale=(0.02, 0.15), ratio=(0.3, 3.3), value=0.0),
            ]
        )

    def __call__(self, image):
        return self.pipeline(image)


def _one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    return torch.nn.functional.one_hot(labels, num_classes=num_classes).float()


def mixup(batch_images: torch.Tensor, batch_labels: torch.Tensor, alpha: float = 0.2):
    if alpha <= 0:
        return batch_images, batch_labels
    batch_size = batch_images.size(0)
    lam = torch.distributions.Beta(alpha, alpha).sample((batch_size,)).to(batch_images.device)
    lam_image = lam.view(batch_size, 1, 1, 1)
    indices = torch.randperm(batch_size, device=batch_images.device)
    mixed_images = batch_images * lam_image + batch_images[indices] * (1.0 - lam_image)
    mixed_labels = batch_labels * lam.view(batch_size, 1) + batch_labels[indices] * (1.0 - lam.view(batch_size, 1))
    return mixed_images, mixed_labels


def cutmix(batch_images: torch.Tensor, batch_labels: torch.Tensor, alpha: float = 0.35):
    if alpha <= 0:
        return batch_images, batch_labels
    batch_size, _, height, width = batch_images.shape
    indices = torch.randperm(batch_size, device=batch_images.device)
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    cut_ratio = torch.sqrt(torch.tensor(1.0 - lam, device=batch_images.device))
    cut_h = int(height * cut_ratio.item())
    cut_w = int(width * cut_ratio.item())
    cy = torch.randint(0, height, (1,), device=batch_images.device).item()
    cx = torch.randint(0, width, (1,), device=batch_images.device).item()
    y1 = max(cy - cut_h // 2, 0)
    y2 = min(cy + cut_h // 2, height)
    x1 = max(cx - cut_w // 2, 0)
    x2 = min(cx + cut_w // 2, width)

    mixed_images = batch_images.clone()
    mixed_images[:, :, y1:y2, x1:x2] = batch_images[indices, :, y1:y2, x1:x2]
    area = ((y2 - y1) * (x2 - x1)) / float(height * width)
    mixed_labels = batch_labels * (1.0 - area) + batch_labels[indices] * area
    return mixed_images, mixed_labels
