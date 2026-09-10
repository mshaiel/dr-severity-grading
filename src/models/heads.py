"""Classification and CORAL Ordinal Regression Heads."""

import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """Standard multi-class classification head with dropout and linear layer."""

    def __init__(
        self,
        in_features: int,
        num_classes: int = 5,
        drop_rate: float = 0.2,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=drop_rate)
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Output unnormalized logits of shape (batch_size, num_classes)."""
        x = self.dropout(x)
        return self.fc(x)


class CoralHead(nn.Module):
    """CORAL Ordinal Regression Head with shared weight vector and per-threshold biases.

    Reference:
        Cao, Mirjalili, Raschka (2020).
        Consistent Rank Logits for Ordinal Regression.

    Architecture:
        A single shared weight vector W in R^{D x 1} projects input features x
        to a scalar logit: s = W^T x.
        Then, K-1 independent bias terms b_k (for k in 1..K-1) are added:
            z_k = s + b_k = W^T x + b_k
        This architectural weight-sharing strictly guarantees that the direction
        in feature space for all threshold boundaries is identical.
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int = 5,
        drop_rate: float = 0.2,
    ) -> None:
        """Initialize CoralHead.

        Args:
            in_features: Dimensionality of backbone feature embedding.
            num_classes: Total ordinal classes K (K-1 binary thresholds).
            drop_rate: Dropout probability.
        """
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.num_thresholds = num_classes - 1

        self.dropout = nn.Dropout(p=drop_rate)
        # Shared linear weight without bias: maps (N, in_features) -> (N, 1)
        self.fc = nn.Linear(in_features, 1, bias=False)
        # K-1 independent learnable biases
        self.biases = nn.Parameter(torch.zeros(self.num_thresholds))

        # Initialize biases to descending or zero values
        nn.init.zeros_(self.biases)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Feature tensor of shape (N, in_features).

        Returns:
            Threshold logits of shape (N, K - 1), where logits[:, k] = W^T x + b_k.
        """
        x = self.dropout(x)
        # scalar_logit shape: (N, 1)
        scalar_logit = self.fc(x)
        # Broadcast add biases shape (K - 1,): yields (N, K - 1)
        logits = scalar_logit + self.biases
        return logits
