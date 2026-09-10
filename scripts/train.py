"""Hydra-driven model training entry point for Diabetic Retinopathy Grading."""

import os
import hydra
from omegaconf import DictConfig, OmegaConf
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import torch
from torch.utils.data import DataLoader

from src.data.aptos_dataset import APTOSDataset
from src.data.samplers import create_weighted_sampler
from src.data.transforms import get_train_transforms, get_val_transforms
from src.losses.coral import CoralLoss
from src.losses.cross_entropy import WeightedCrossEntropyLoss
from src.losses.focal import FocalLoss
from src.models.model_factory import build_model
from src.training.early_stopping import EarlyStopping
from src.training.scheduler import build_scheduler
from src.training.trainer import Trainer
from src.utils.logging_utils import get_logger, setup_logging
from src.utils.seed import seed_everything

logger = get_logger("TrainScript")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Execute training pipeline with Hydra config."""
    setup_logging()
    logger.info("Initializing training run with configuration:\n" + OmegaConf.to_yaml(cfg))

    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using compute device: {device}")

    # Initialize W&B if requested and available
    if cfg.wandb.enabled:
        try:
            import wandb

            wandb.init(
                project=cfg.wandb.project,
                entity=cfg.wandb.entity,
                name=cfg.wandb.name,
                tags=cfg.wandb.tags,
                config=OmegaConf.to_container(cfg, resolve=True),
            )
            logger.info("Weights & Biases tracking initialized.")
        except Exception as e:
            logger.warning(f"Failed to initialize W&B: {e}. Proceeding without W&B.")

    # 1. Load data CSV and build stratified 80/20 train/val split
    train_csv_path = cfg.data.train_csv
    if not os.path.exists(train_csv_path):
        raise FileNotFoundError(f"Training CSV not found at {train_csv_path}")

    df = pd.read_csv(train_csv_path)
    label_col = "diagnosis" if "diagnosis" in df.columns else df.columns[1]

    skf = StratifiedKFold(n_splits=int(1.0 / cfg.data.val_split), shuffle=True, random_state=cfg.seed)
    train_idx, val_idx = next(skf.split(df, df[label_col]))

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    logger.info(f"Dataset split: {len(train_df)} train samples, {len(val_df)} val samples.")

    # 2. Build datasets and dataloaders
    train_transforms = get_train_transforms(image_size=cfg.data.image_size)
    val_transforms = get_val_transforms(image_size=cfg.data.image_size)

    images_dir = cfg.data.preprocessed_dir if cfg.data.preprocessed else cfg.data.images_dir

    train_dataset = APTOSDataset(
        df_or_csv=train_df,
        images_dir=images_dir,
        transform=train_transforms,
        image_size=cfg.data.image_size,
        apply_ben_graham=not cfg.data.preprocessed,
        is_preprocessed=cfg.data.preprocessed,
    )

    val_dataset = APTOSDataset(
        df_or_csv=val_df,
        images_dir=images_dir,
        transform=val_transforms,
        image_size=cfg.data.image_size,
        apply_ben_graham=not cfg.data.preprocessed,
        is_preprocessed=cfg.data.preprocessed,
    )

    sampler = None
    shuffle = True
    if getattr(cfg.data, "use_weighted_sampler", False):
        sampler = create_weighted_sampler(train_df[label_col].values)
        shuffle = False
        logger.info("Configured WeightedRandomSampler for class imbalance.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory and torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory and torch.cuda.is_available(),
    )

    # 3. Build Model
    model = build_model(model_config=cfg.model, loss_name=cfg.loss.name)
    logger.info(f"Instantiated model {cfg.model.name} with {cfg.loss.name} head.")

    # 4. Build Loss Criterion
    is_ordinal = (cfg.loss.name.lower() == "coral")
    class_weights = train_dataset.get_class_weights().to(device)

    if cfg.loss.name.lower() == "coral":
        criterion = CoralLoss(num_classes=cfg.model.num_classes)
    elif cfg.loss.name.lower() == "focal":
        weights = class_weights if cfg.loss.use_class_weights else None
        criterion = FocalLoss(gamma=cfg.loss.gamma, alpha=weights)
    elif cfg.loss.name.lower() == "cross_entropy":
        weights = class_weights if cfg.loss.use_class_weights else None
        criterion = WeightedCrossEntropyLoss(weight=weights)
    else:
        raise ValueError(f"Unknown loss: {cfg.loss.name}")

    # 5. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )

    scheduler = build_scheduler(
        optimizer=optimizer,
        scheduler_name=cfg.training.scheduler,
        t_0=cfg.training.t_0,
        t_mult=cfg.training.t_mult,
        min_lr=cfg.training.min_lr,
    )

    early_stopping = EarlyStopping(
        patience=cfg.training.early_stopping.patience,
        min_delta=cfg.training.early_stopping.min_delta,
        mode=cfg.training.early_stopping.mode,
    )

    # 6. Train
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=cfg.training.epochs,
        gradient_clip_val=cfg.training.gradient_clip_val,
        early_stopping=early_stopping,
        checkpoint_dir=cfg.training.checkpoint.dirpath,
        checkpoint_filename=cfg.training.checkpoint.filename,
        is_ordinal=is_ordinal,
        use_wandb=cfg.wandb.enabled,
    )

    history = trainer.fit(train_loader=train_loader, val_loader=val_loader)
    logger.info("Training complete.")

    os.makedirs("results/tables", exist_ok=True)
    best_metrics = history.get("best_metrics", {})
    if best_metrics:
        record = {k: v for k, v in best_metrics.items() if not isinstance(v, list)}
        record["model"] = cfg.model.name
        record["loss"] = cfg.loss.name
        out_csv = f"results/tables/{cfg.model.name}_{cfg.loss.name}.csv"
        pd.DataFrame([record]).to_csv(out_csv, index=False)
        logger.info(f"Saved best evaluation metrics to {out_csv}")


if __name__ == "__main__":
    main()
