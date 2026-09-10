"""Early Stopping tracker monitoring validation metrics."""

from typing import Optional


class EarlyStopping:
    """Early stops the training if monitored metric doesn't improve after patience epochs."""

    def __init__(
        self,
        patience: int = 7,
        min_delta: float = 0.001,
        mode: str = "max",
    ) -> None:
        """Initialize EarlyStopping.

        Args:
            patience: Number of epochs to wait after last improvement.
            min_delta: Minimum change in the monitored quantity to qualify as improvement.
            mode: One of 'min' or 'max'. For QWK, use 'max'.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode.lower()
        if self.mode not in ["min", "max"]:
            raise ValueError(f"Mode must be 'min' or 'max', got {mode}")

        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False

    def __call__(self, current_score: float) -> bool:
        """Update tracker with current score.

        Args:
            current_score: Current epoch score for the monitored metric.

        Returns:
            is_best (bool): True if this score is the best observed so far.
        """
        if self.best_score is None:
            self.best_score = current_score
            return True

        if self.mode == "max":
            improved = current_score > (self.best_score + self.min_delta)
        else:
            improved = current_score < (self.best_score - self.min_delta)

        if improved:
            self.best_score = current_score
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False
