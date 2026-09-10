"""Post-hoc Temperature Scaling calibration script."""

import os
import hydra
from omegaconf import DictConfig
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import torch
from torch.utils.data import DataLoader

from src.data.aptos_dataset import APTOSDataset
from src.data.transforms import get_val_transforms
from src.evaluation.calibration import TemperatureScaler, plot_reliability_diagram
from src.evaluation.metrics import compute_ece
from src.losses.coral import coral_probabilities
from src.models.model_factory import build_model
from src.utils.checkpoint import load_checkpoint
from src.utils.logging_utils import get_logger, setup_logging

logger = get_logger("CalibrateScript")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_logging()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = cfg.inference.checkpoint_path or os.path.join(
        cfg.training.checkpoint.dirpath, cfg.training.checkpoint.filename
    )
    logger.info(f"Calibrating model from: {checkpoint_path}")

    model = build_model(model_config=cfg.model, loss_name=cfg.loss.name)
    checkpoint = load_checkpoint(checkpoint_path, device=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    is_ordinal = (cfg.loss.name.lower() == "coral")

    # Load validation split
    df = pd.read_csv(cfg.data.train_csv)
    label_col = "diagnosis" if "diagnosis" in df.columns else df.columns[1]
    skf = StratifiedKFold(n_splits=int(1.0 / cfg.data.val_split), shuffle=True, random_state=cfg.seed)
    _, val_idx = next(skf.split(df, df[label_col]))
    val_df = df.iloc[val_idx].reset_index(drop=True)

    val_dataset = APTOSDataset(
        df_or_csv=val_df,
        images_dir=cfg.data.images_dir,
        transform=get_val_transforms(image_size=cfg.data.image_size),
        image_size=cfg.data.image_size,
    )
    val_loader = DataLoader(val_dataset, batch_size=cfg.data.batch_size, shuffle=False)

    all_logits = []
    all_targets = []

    with torch.no_grad():
        for images, targets, _ in val_loader:
            images = images.to(device)
            logits = model(images)
            all_logits.append(logits)
            all_targets.append(targets)

    val_logits = torch.cat(all_logits, dim=0)
    val_targets = torch.cat(all_targets, dim=0)

    # Initial probabilities and ECE
    if is_ordinal:
        init_probs = coral_probabilities(val_logits)
    else:
        init_probs = torch.softmax(val_logits, dim=1)

    init_ece, _ = compute_ece(init_probs, val_targets)
    logger.info(f"Uncalibrated Validation ECE: {init_ece:.4f}")

    # Optimize Temperature
    scaler = TemperatureScaler()
    optimal_t = scaler.fit(val_logits, val_targets, is_ordinal=is_ordinal)
    logger.info(f"Optimal Temperature Parameter T: {optimal_t:.4f}")

    # Post-calibration evaluation
    scaled_logits = scaler(val_logits)
    if is_ordinal:
        cal_probs = coral_probabilities(scaled_logits)
    else:
        cal_probs = torch.softmax(scaled_logits, dim=1)

    cal_ece, _ = compute_ece(cal_probs, val_targets)
    logger.info(f"Calibrated Validation ECE: {cal_ece:.4f} (Delta: {cal_ece - init_ece:.4f})")

    # Plot and save reliability diagram
    os.makedirs("results/figures", exist_ok=True)
    fig = plot_reliability_diagram(
        probs=cal_probs,
        y_true=val_targets,
        title=f"Calibrated Reliability Diagram ({cfg.model.name} {cfg.loss.name})",
    )
    fig_path = f"results/figures/calibration_{cfg.model.name}_{cfg.loss.name}.png"
    fig.savefig(fig_path, dpi=300)
    logger.info(f"Saved reliability diagram to {fig_path}")


if __name__ == "__main__":
    main()
