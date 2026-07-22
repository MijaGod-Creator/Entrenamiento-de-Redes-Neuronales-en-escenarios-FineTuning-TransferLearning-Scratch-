from pathlib import Path

import pandas as pd
import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

from src.config.settings import IMAGE_SIZE

ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_split_csv(csv_path: Path) -> tuple[list[str], list[int]]:
    frame = pd.read_csv(csv_path)
    return frame["path"].astype(str).tolist(), frame["label"].astype(int).tolist()


def build_transform(
    image_size: tuple[int, int] = IMAGE_SIZE,
    train: bool = False,
    use_augmentation: bool = False,
):
    normalize = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    base = [T.Resize(image_size, interpolation=InterpolationMode.BILINEAR)]

    if train and use_augmentation:
        transforms = [
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(15, interpolation=InterpolationMode.BILINEAR, fill=0),
            T.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.88, 1.12), interpolation=InterpolationMode.BILINEAR, fill=0),
            T.RandomResizedCrop(image_size, scale=(0.78, 1.0), ratio=(0.9, 1.1), interpolation=InterpolationMode.BILINEAR),
            T.ColorJitter(brightness=0.15, contrast=0.15),
        ]
    else:
        transforms = base

    transforms.extend(
        [
            T.ToTensor(),
            normalize,
        ]
    )

    if train and use_augmentation:
        transforms.append(T.RandomErasing(p=0.3, scale=(0.02, 0.15), ratio=(0.3, 3.3), value=0.0))

    return T.Compose(transforms)


class EmotionDataset(Dataset):
    def __init__(self, csv_path: Path, image_size: tuple[int, int] = IMAGE_SIZE, train: bool = False, use_augmentation: bool = False):
        self.paths, self.labels = load_split_csv(csv_path)
        self.transform = build_transform(image_size=image_size, train=train, use_augmentation=use_augmentation)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        image = Image.open(self.paths[index]).convert("RGB")
        image = self.transform(image)
        label = torch.tensor(self.labels[index], dtype=torch.long)
        return image, label


def build_dataset(
    csv_path: Path,
    batch_size: int,
    image_size: tuple[int, int] = IMAGE_SIZE,
    shuffle: bool = False,
    use_augmentation: bool = False,
    train: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    seed: int = 42,
) -> DataLoader:
    dataset = EmotionDataset(csv_path, image_size=image_size, train=train, use_augmentation=use_augmentation)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        generator=generator if shuffle else None,
        persistent_workers=num_workers > 0,
    )


def build_dataset_bundle(
    processed_dir: Path,
    batch_size: int,
    image_size: tuple[int, int] = IMAGE_SIZE,
    use_augmentation: bool = False,
    seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> dict[str, DataLoader]:
    return {
        "train": build_dataset(
            processed_dir / "train.csv",
            batch_size=batch_size,
            image_size=image_size,
            shuffle=True,
            use_augmentation=use_augmentation,
            train=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            seed=seed,
        ),
        "validation": build_dataset(
            processed_dir / "validation.csv",
            batch_size=batch_size,
            image_size=image_size,
            shuffle=False,
            train=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            seed=seed,
        ),
        "test": build_dataset(
            processed_dir / "test.csv",
            batch_size=batch_size,
            image_size=image_size,
            shuffle=False,
            train=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            seed=seed,
        ),
    }
