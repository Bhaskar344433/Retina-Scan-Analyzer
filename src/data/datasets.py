"""
Dataset loading and preprocessing for retinal image analysis.
Supports classification (RetinaMNIST) and segmentation (DRIVE) datasets.
"""

import os
from dataclasses import dataclass
from typing import Optional, Tuple, Callable
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


@dataclass
class DatasetConfig:
    """Configuration for dataset loading"""
    name: str
    split: str
    subset: Optional[str] = None
    cache_dir: Optional[str] = "./data/hf-cache"


class RetinalDataset(Dataset):
    """
    PyTorch Dataset wrapper for retinal images with augmentation support.

    Args:
        hf_dataset: HuggingFace dataset object
        transform: Albumentations transform pipeline
        mode: 'classification' or 'segmentation'
    """

    def __init__(
        self,
        hf_dataset,
        transform: Optional[Callable] = None,
        mode: str = 'classification'
    ):
        self.dataset = hf_dataset
        self.transform = transform
        self.mode = mode

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.dataset[idx]

        if self.mode == 'classification':
            # RetinaMNIST format
            image = np.array(item['image'])
            label = item['label']

            if self.transform:
                augmented = self.transform(image=image)
                image = augmented['image']
            else:
                image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

            return image, torch.tensor(label, dtype=torch.long)

        elif self.mode == 'segmentation':
            # DRIVE format
            image = np.array(item['image'])
            mask = np.array(item['mask'])

            if self.transform:
                augmented = self.transform(image=image, mask=mask)
                image = augmented['image']
                mask = augmented['mask']
            else:
                image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
                mask = torch.from_numpy(mask).unsqueeze(0).float() / 255.0

            return image, mask


def get_classification_transforms(
    image_size: int = 224,
    is_training: bool = True
) -> A.Compose:
    """
    Get augmentation pipeline for classification tasks.

    Args:
        image_size: Target image size
        is_training: If True, apply training augmentations

    Returns:
        Albumentations composition
    """
    if is_training:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.5),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0)),
                A.GaussianBlur(blur_limit=(3, 7)),
            ], p=0.3),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20),
            ], p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])


def get_segmentation_transforms(
    image_size: int = 512,
    is_training: bool = True
) -> A.Compose:
    """
    Get augmentation pipeline for segmentation tasks.

    Args:
        image_size: Target image size
        is_training: If True, apply training augmentations

    Returns:
        Albumentations composition
    """
    if is_training:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.3),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 30.0)),
                A.GaussianBlur(blur_limit=(3, 5)),
            ], p=0.2),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])


def load_classification_dataset(
    config: DatasetConfig,
    transform: Optional[Callable] = None
) -> RetinalDataset:
    """
    Load classification dataset (e.g., RetinaMNIST).

    Args:
        config: Dataset configuration
        transform: Optional transform pipeline

    Returns:
        RetinalDataset instance
    """
    print(f"Loading classification dataset: {config.name}/{config.subset} - {config.split}")

    # Load from HuggingFace
    if config.subset:
        hf_dataset = load_dataset(
            config.name,
            config.subset,
            split=config.split,
            cache_dir=config.cache_dir,
            trust_remote_code=True
        )
    else:
        hf_dataset = load_dataset(
            config.name,
            split=config.split,
            cache_dir=config.cache_dir,
            trust_remote_code=True
        )

    return RetinalDataset(hf_dataset, transform=transform, mode='classification')


def load_segmentation_dataset(
    config: DatasetConfig,
    transform: Optional[Callable] = None
) -> RetinalDataset:
    """
    Load segmentation dataset (e.g., DRIVE).

    Args:
        config: Dataset configuration
        transform: Optional transform pipeline

    Returns:
        RetinalDataset instance
    """
    print(f"Loading segmentation dataset: {config.name} - {config.split}")

    # Load from HuggingFace
    hf_dataset = load_dataset(
        config.name,
        split=config.split,
        cache_dir=config.cache_dir,
        trust_remote_code=True
    )

    return RetinalDataset(hf_dataset, transform=transform, mode='segmentation')
