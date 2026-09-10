"""Model assembly and factory functions."""

from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.backbone import EfficientNetBackbone
from src.models.heads import ClassificationHead, CoralHead
from src.losses.coral import coral_predict, coral_probabilities


class DRSeverityModel(nn.Module):
    """Unified Diabetic Retinopathy grading model wrapping backbone and swappable head."""

    def __init__(
        self,
        backbone: nn.Module,
        head: nn.Module,
        is_ordinal: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.is_ordinal = is_ordinal

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute logits from input images.

        Args:
            x: Input images of shape (N, C, H, W).

        Returns:
            Logits of shape (N, num_classes) for CE/Focal, or (N, num_classes - 1) for CORAL.
        """
        features = self.backbone(x)
        logits = self.head(features)
        return logits

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract intermediate spatial feature maps (for Grad-CAM)."""
        return self.backbone.forward_features(x)

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        threshold: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inference helper computing predicted integer grades and categorical probabilities.

        Args:
            x: Input tensor of shape (N, C, H, W).
            threshold: Probability threshold for CORAL rank decoding.

        Returns:
            Tuple of (grades, class_probabilities):
                grades: Integer tensor of shape (N,)
                class_probabilities: Tensor of shape (N, num_classes)
        """
        logits = self.forward(x)

        if self.is_ordinal:
            grades = coral_predict(logits, threshold=threshold)
            probs = coral_probabilities(logits)
        else:
            probs = F.softmax(logits, dim=1)
            grades = torch.argmax(probs, dim=1)

        return grades, probs


def build_model(
    model_config: Union[Dict[str, Any], Any],
    loss_name: str = "coral",
) -> DRSeverityModel:
    """Instantiate a DRSeverityModel from configuration.

    Args:
        model_config: Dictionary or OmegaConf object containing model specifications.
        loss_name: Name of the loss function ('coral', 'cross_entropy', 'focal').

    Returns:
        Configured DRSeverityModel instance.
    """
    backbone_name = getattr(model_config, "backbone", "efficientnet_b0")
    pretrained = getattr(model_config, "pretrained", True)
    drop_rate = getattr(model_config, "drop_rate", 0.2)
    drop_path_rate = getattr(model_config, "drop_path_rate", 0.1)
    num_classes = getattr(model_config, "num_classes", 5)

    backbone = EfficientNetBackbone(
        model_name=backbone_name,
        pretrained=pretrained,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
    )

    is_ordinal = (loss_name.lower() == "coral")

    if is_ordinal:
        head = CoralHead(
            in_features=backbone.num_features,
            num_classes=num_classes,
            drop_rate=drop_rate,
        )
    else:
        head = ClassificationHead(
            in_features=backbone.num_features,
            num_classes=num_classes,
            drop_rate=drop_rate,
        )

    return DRSeverityModel(
        backbone=backbone,
        head=head,
        is_ordinal=is_ordinal,
    )
