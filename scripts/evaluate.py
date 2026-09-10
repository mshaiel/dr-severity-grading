"""Offline evaluation script for APTOS and Messidor-2 datasets."""

import os
import hydra
from omegaconf import DictConfig
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.aptos_dataset import APTOSDataset
from src.data.messidor_dataset import Messidor2Dataset
from src.data.transforms import get_val_transforms
from src.evaluation.domain_shift import evaluate_domain_shift
from src.models.model_factory import build_model
from src.utils.checkpoint import load_checkpoint
from src.utils.logging_utils import get_logger, setup_logging

logger = get_logger("EvaluateScript")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_logging()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = cfg.inference.checkpoint_path or os.path.join(
        cfg.training.checkpoint.dirpath, cfg.training.checkpoint.filename
    )
    logger.info(f"Loading checkpoint from: {checkpoint_path}")

    # Build model and load weights
    model = build_model(model_config=cfg.model, loss_name=cfg.loss.name)
    checkpoint = load_checkpoint(checkpoint_path, device=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    is_ordinal = (cfg.loss.name.lower() == "coral")
    val_transforms = get_val_transforms(image_size=cfg.data.image_size)

    # Determine dataset to evaluate
    if cfg.data.name == "messidor2":
        logger.info("Setting up Messidor-2 Out-Of-Distribution evaluation...")
        dataset = Messidor2Dataset(
            df_or_csv=cfg.data.csv_path,
            images_dir=cfg.data.images_dir,
            transform=val_transforms,
            image_size=cfg.data.image_size,
            filter_gradable=cfg.data.filter_adjudicated_gradable,
        )
    else:
        logger.info("Setting up APTOS validation split evaluation...")
        from sklearn.model_selection import StratifiedKFold
        df = pd.read_csv(cfg.data.train_csv)
        label_col = "diagnosis" if "diagnosis" in df.columns else df.columns[1]
        skf = StratifiedKFold(n_splits=int(1.0 / cfg.data.val_split), shuffle=True, random_state=cfg.seed)
        _, val_idx = next(skf.split(df, df[label_col]))
        val_df = df.iloc[val_idx].reset_index(drop=True)

        dataset = APTOSDataset(
            df_or_csv=val_df,
            images_dir=cfg.data.images_dir,
            transform=val_transforms,
            image_size=cfg.data.image_size,
        )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
    )

    use_tta = getattr(cfg.inference, "tta", False)
    logger.info(f"Evaluating {len(dataset)} samples. TTA: {use_tta}")

    metrics = evaluate_domain_shift(
        model=model,
        dataloader=dataloader,
        device=device,
        use_tta=use_tta,
        is_ordinal=is_ordinal,
    )

    logger.info("=== Evaluation Results ===")
    logger.info(f"Dataset: {cfg.data.name} | Model: {cfg.model.name} | Loss: {cfg.loss.name}")
    logger.info(f"QWK Score: {metrics['qwk']:.4f}")
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    logger.info(f"Macro F1: {metrics['f1_macro']:.4f}")
    if "ece" in metrics:
        logger.info(f"Expected Calibration Error (ECE): {metrics['ece']:.4f}")

    # Save results table
    os.makedirs("results/tables", exist_ok=True)
    out_csv = f"results/tables/{cfg.data.name}_{cfg.model.name}_{cfg.loss.name}{'_tta' if use_tta else ''}.csv"
    record = {k: v for k, v in metrics.items() if not isinstance(v, list)}
    pd.DataFrame([record]).to_csv(out_csv, index=False)
    logger.info(f"Saved evaluation metrics to: {out_csv}")


if __name__ == "__main__":
    main()
