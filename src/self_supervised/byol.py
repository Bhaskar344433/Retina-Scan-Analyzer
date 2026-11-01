"""
BYOL (Bootstrap Your Own Latent)
Implementation for retinal image pretraining.

Reference: Grill et al., 2020 - https://arxiv.org/abs/2006.07733
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import copy


class BYOLAugmentation:
    """
    Data augmentation pipeline for BYOL.
    Similar to SimCLR but with different augmentation parameters.
    """

    def __init__(self, image_size: int = 224):
        self.transform1 = A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8),
            A.GaussianBlur(blur_limit=(3, 7), p=0.1),
            A.Solarize(threshold=128, p=0.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

        self.transform2 = A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8),
            A.GaussianBlur(blur_limit=(3, 7), p=0.5),
            A.Solarize(threshold=128, p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

    def __call__(self, image: np.ndarray):
        """Generate two augmented views"""
        view1 = self.transform1(image=image)['image']
        view2 = self.transform2(image=image)['image']
        return view1, view2


class MLPHead(nn.Module):
    """MLP projection and prediction heads for BYOL"""

    def __init__(self, in_dim: int, hidden_dim: int = 4096, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)


class BYOL(nn.Module):
    """
    BYOL model for self-supervised pretraining.

    Args:
        backbone: Encoder network
        feature_dim: Dimension of backbone output
        projection_dim: Dimension of projection head output
        hidden_dim: Hidden dimension in projection head
        moving_average_decay: EMA decay rate for target network
    """

    def __init__(
        self,
        backbone: nn.Module,
        feature_dim: int,
        projection_dim: int = 256,
        hidden_dim: int = 4096,
        moving_average_decay: float = 0.996
    ):
        super().__init__()

        # Online network
        self.online_encoder = backbone
        self.online_projector = MLPHead(feature_dim, hidden_dim, projection_dim)
        self.online_predictor = MLPHead(projection_dim, hidden_dim, projection_dim)

        # Target network (exponential moving average of online network)
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.target_projector = copy.deepcopy(self.online_projector)

        # Stop gradient for target network
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        for param in self.target_projector.parameters():
            param.requires_grad = False

        self.m = moving_average_decay

    @torch.no_grad()
    def update_target_network(self):
        """Update target network with exponential moving average"""
        for online_params, target_params in zip(
            self.online_encoder.parameters(),
            self.target_encoder.parameters()
        ):
            target_params.data = self.m * target_params.data + (1 - self.m) * online_params.data

        for online_params, target_params in zip(
            self.online_projector.parameters(),
            self.target_projector.parameters()
        ):
            target_params.data = self.m * target_params.data + (1 - self.m) * online_params.data

    def forward(self, x1, x2):
        """
        Args:
            x1, x2: Two augmented views

        Returns:
            Online and target predictions
        """
        # Online network predictions
        online_repr_1 = self.online_encoder(x1)
        online_proj_1 = self.online_projector(online_repr_1)
        online_pred_1 = self.online_predictor(online_proj_1)

        online_repr_2 = self.online_encoder(x2)
        online_proj_2 = self.online_projector(online_repr_2)
        online_pred_2 = self.online_predictor(online_proj_2)

        # Target network projections (no gradient)
        with torch.no_grad():
            target_repr_1 = self.target_encoder(x1)
            target_proj_1 = self.target_projector(target_repr_1)

            target_repr_2 = self.target_encoder(x2)
            target_proj_2 = self.target_projector(target_repr_2)

        return online_pred_1, online_pred_2, target_proj_1.detach(), target_proj_2.detach()

    @staticmethod
    def regression_loss(x, y):
        """
        Cosine similarity loss for BYOL.

        Args:
            x: Online predictions
            y: Target projections (detached)

        Returns:
            Loss value
        """
        x = F.normalize(x, dim=-1, p=2)
        y = F.normalize(y, dim=-1, p=2)
        return 2 - 2 * (x * y).sum(dim=-1).mean()

    def compute_loss(self, online_pred_1, online_pred_2, target_proj_1, target_proj_2):
        """
        Compute symmetric loss for both views.

        Args:
            online_pred_1, online_pred_2: Predictions from online network
            target_proj_1, target_proj_2: Projections from target network

        Returns:
            Total loss
        """
        loss1 = self.regression_loss(online_pred_1, target_proj_2)
        loss2 = self.regression_loss(online_pred_2, target_proj_1)
        return (loss1 + loss2) / 2


def train_byol_epoch(
    model: BYOL,
    dataloader,
    optimizer,
    device: torch.device
):
    """
    Train BYOL for one epoch.

    Args:
        model: BYOL model
        dataloader: DataLoader
        optimizer: Optimizer
        device: Device to train on

    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0
    num_batches = 0

    for batch in dataloader:
        images = batch[0].to(device)

        # Create two views
        batch_size = images.shape[0] // 2
        x1 = images[:batch_size]
        x2 = images[batch_size:]

        # Forward pass
        online_pred_1, online_pred_2, target_proj_1, target_proj_2 = model(x1, x2)

        # Compute loss
        loss = model.compute_loss(online_pred_1, online_pred_2, target_proj_1, target_proj_2)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update target network
        model.update_target_network()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


if __name__ == "__main__":
    # Test BYOL
    from torchvision.models import resnet50

    # Create backbone
    backbone = resnet50(pretrained=False)
    backbone.fc = nn.Identity()

    # Create BYOL model
    model = BYOL(backbone, feature_dim=2048, projection_dim=256)

    # Test forward pass
    x1 = torch.randn(4, 3, 224, 224)
    x2 = torch.randn(4, 3, 224, 224)
    online_pred_1, online_pred_2, target_proj_1, target_proj_2 = model(x1, x2)

    print(f"Online pred 1 shape: {online_pred_1.shape}")
    print(f"Target proj 1 shape: {target_proj_1.shape}")

    # Test loss
    loss = model.compute_loss(online_pred_1, online_pred_2, target_proj_1, target_proj_2)
    print(f"Loss: {loss.item():.4f}")
