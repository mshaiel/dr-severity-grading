"""Out-of-distribution domain shift evaluation harness for Messidor-2."""

from typing import Any, Dict, Optional
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.evaluation.metrics import compute_metrics, predict_with_tta
from src.losses.coral import coral_predict, coral_probabilities


def evaluate_domain_shift(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device = torch.device("cpu"),
    use_tta: bool = False,
    is_ordinal: bool = True,
) -> Dict[str, Any]:
    """Run full evaluation on held-out out-of-distribution dataset (Messidor-2).

    Args:
        model: Trained PyTorch DRSeverityModel.
        dataloader: DataLoader for Messidor-2.
        device: Device to run evaluation on.
        use_tta: If True, uses horizontal flip Test-Time Augmentation.
        is_ordinal: Whether model uses CORAL ordinal head.

    Returns:
        Dictionary of computed metrics including QWK, ECE, F1 scores, and confusion matrix.
    """
    model.eval()
    model.to(device)

    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for images, targets, _ in tqdm(dataloader, desc="Evaluating Domain Shift"):
            images = images.to(device)

            if use_tta:
                preds, probs = predict_with_tta(model, images, is_ordinal=is_ordinal)
            else:
                logits = model(images)
                if is_ordinal:
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
    return metrics
