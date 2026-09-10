"""Backbone feature extractors supporting EfficientNet variants via timm."""

from typing import Tuple
import timm
import torch
import torch.nn as nn


class EfficientNetBackbone(nn.Module):
    """EfficientNet feature extractor wrapping timm pretrained models."""

    def __init__(
        self,
        model_name: str = "efficientnet_b0",
        pretrained: bool = True,
        drop_rate: float = 0.2,
        drop_path_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.pretrained = pretrained

        # Create model with head removed (num_classes=0 returns pooled feature vector)
        self.encoder = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
        )
        self.num_features = self.encoder.num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract global pooled feature representations.

        Args:
            x: Input tensor of shape (N, C, H, W).

        Returns:
            Feature embedding of shape (N, num_features).
        """
        return self.encoder(x)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract unpooled spatial feature maps before global pooling (for Grad-CAM).

        Args:
            x: Input tensor of shape (N, C, H, W).

        Returns:
            Spatial feature maps of shape (N, C_feat, H_feat, W_feat).
        """
        return self.encoder.forward_features(x)
