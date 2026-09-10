"""Main Model Trainer for Diabetic Retinopathy Severity Grading."""

import os
from typing import Any, Dict, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.evaluation.metrics import compute_metrics
from src.losses.coral import coral_predict, coral_probabilities
from src.training.early_stopping import EarlyStopping
from src.utils.checkpoint import save_checkpoint
from src.utils.logging_utils import get_logger

logger = get_logger("Trainer")


class Trainer:
    """Orchestrates model training, validation, metric calculation, and checkpointing."""

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        device: torch.device = torch.device("cpu"),
        epochs: int = 30,
        gradient_clip_val: float = 1.0,
        early_stopping: Optional[EarlyStopping] = None,
        checkpoint_dir: str = "checkpoints",
        checkpoint_filename: str = "best_model.pth",
        is_ordinal: bool = True,
        use_wandb: bool = False,
    ) -> None:
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.epochs = epochs
        self.gradient_clip_val = gradient_clip_val
        self.early_stopping = early_stopping
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_filename = checkpoint_filename
        self.is_ordinal = is_ordinal
        self.use_wandb = use_wandb

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.best_checkpoint_path = os.path.join(self.checkpoint_dir, self.checkpoint_filename)

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> float:
        """Run one full training epoch."""
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.epochs} [Train]")
        for images, targets, _ in pbar:
            images = images.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, targets)

            loss.backward()
            if self.gradient_clip_val > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)

            self.optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        return total_loss / max(len(dataloader), 1)

    @torch.no_grad()
    def validate_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, Any]:
        """Run validation and compute QWK, ECE, F1, and loss."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        all_probs = []

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.epochs} [Val]")
        for images, targets, _ in pbar:
            images = images.to(self.device)
            targets_gpu = targets.to(self.device)

            logits = self.model(images)
            loss = self.criterion(logits, targets_gpu)
            total_loss += loss.item()

            if self.is_ordinal:
                preds = coral_predict(logits)
                probs = coral_probabilities(logits)
            else:
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.numpy())
            all_probs.append(probs.cpu().numpy())

        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_targets)
        prob_arr = np.concatenate(all_probs)

        metrics = compute_metrics(y_true, y_pred, probs=prob_arr)
        metrics["loss"] = total_loss / max(len(dataloader), 1)
        return metrics

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, Any]:
        """Run complete training cycle across all epochs with early stopping."""
        best_qwk = -1.0
        history: Dict[str, list] = {
            "train_loss": [],
            "val_loss": [],
            "val_qwk": [],
            "val_ece": [],
        }

        for epoch in range(self.epochs):
            train_loss = self.train_epoch(train_loader, epoch)
            val_metrics = self.validate_epoch(val_loader, epoch)

            val_loss = val_metrics["loss"]
            val_qwk = val_metrics["qwk"]
            val_ece = val_metrics.get("ece", 0.0)

            if self.scheduler is not None:
                self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]

            logger.info(
                f"Epoch {epoch+1:02d}/{self.epochs:02d} | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Val QWK: {val_qwk:.4f} | Val ECE: {val_ece:.4f} | LR: {current_lr:.6f}"
            )

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_qwk"].append(val_qwk)
            history["val_ece"].append(val_ece)

            # Log to W&B if active
            if self.use_wandb:
                try:
                    import wandb

                    if wandb.run is not None:
                        log_dict = {
                            "epoch": epoch + 1,
                            "train/loss": train_loss,
                            "val/loss": val_loss,
                            "val/qwk": val_qwk,
                            "val/accuracy": val_metrics["accuracy"],
                            "val/balanced_acc": val_metrics["balanced_accuracy"],
                            "val/f1_macro": val_metrics["f1_macro"],
                            "val/ece": val_ece,
                            "lr": current_lr,
                        }
                        for c in range(5):
                            log_dict[f"val/f1_class_{c}"] = val_metrics.get(f"f1_class_{c}", 0.0)
                        wandb.log(log_dict)
                except ImportError:
                    pass

            # Checkpoint best model
            is_best = val_qwk > best_qwk
            if is_best:
                best_qwk = val_qwk
                self.best_metrics = val_metrics
                logger.info(f"⭐ New best validation QWK: {best_qwk:.4f}. Saving checkpoint.")
                save_checkpoint(
                    state={
                        "epoch": epoch + 1,
                        "state_dict": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "best_qwk": best_qwk,
                        "val_metrics": val_metrics,
                    },
                    filepath=os.path.join(self.checkpoint_dir, f"epoch_{epoch+1}.pth"),
                    is_best=True,
                    best_filepath=self.best_checkpoint_path,
                )

            # Early stopping check
            if self.early_stopping is not None:
                self.early_stopping(val_qwk)
                if self.early_stopping.early_stop:
                    logger.info(f"Early stopping triggered at epoch {epoch+1}. Halting.")
                    break

        history["best_metrics"] = getattr(self, "best_metrics", {})
        return history
