import cv2
import numpy as np

try:
    import albumentations as A
except ImportError:  # pragma: no cover
    A = None


def build_albumentations_pipeline(image_size: tuple[int, int]):
    if A is None:
        raise ImportError("Albumentations no esta instalado. Ejecute pip install albumentations.")
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, border_mode=cv2.BORDER_REFLECT_101, p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.08,
                scale_limit=0.12,
                rotate_limit=0,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.4,
            ),
            A.RandomResizedCrop(size=image_size, scale=(0.78, 1.0), ratio=(0.9, 1.1), p=0.35),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),
            A.GaussNoise(var_limit=(5.0, 25.0), p=0.25),
            A.Blur(blur_limit=3, p=0.2),
            A.CoarseDropout(max_holes=1, max_height=32, max_width=32, fill_value=0, p=0.3),
        ]
    )


def apply_albumentations(image: np.ndarray, pipeline) -> np.ndarray:
    image_uint8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    augmented = pipeline(image=image_uint8)["image"]
    return augmented.astype(np.float32) / 255.0
