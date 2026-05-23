"""
model/tft.py — TFT 모델 및 PyTorch Lightning Trainer 설정
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytorch_lightning as pl
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss, MAE, RMSE
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import CSVLogger

from config import (
    HIDDEN_SIZE, ATTENTION_HEAD_SIZE, DROPOUT,
    HIDDEN_CONTINUOUS_SIZE, LEARNING_RATE, QUANTILES,
    MAX_EPOCHS, PATIENCE, GRADIENT_CLIP,
    CHECKPOINT_DIR, LOG_DIR,
)


def build_model(train_dataset):
    """TimeSeriesDataSet으로부터 TFT 모델 생성."""
    model = TemporalFusionTransformer.from_dataset(
        train_dataset,
        learning_rate          = LEARNING_RATE,
        hidden_size            = HIDDEN_SIZE,
        attention_head_size    = ATTENTION_HEAD_SIZE,
        dropout                = DROPOUT,
        hidden_continuous_size = HIDDEN_CONTINUOUS_SIZE,
        loss                   = QuantileLoss(quantiles=QUANTILES),
        log_interval           = 10,
        reduce_on_plateau_patience = 4,
        logging_metrics        = [MAE(), RMSE()],
    )
    print(f"[model] Parameters: {model.size() / 1e3:.1f}k")
    return model


def build_trainer(run_name: str = "tft_run") -> pl.Trainer:
    callbacks = [
        EarlyStopping(
            monitor  = "val_loss",
            patience = PATIENCE,
            mode     = "min",
            verbose  = True,
        ),
        ModelCheckpoint(
            dirpath   = str(CHECKPOINT_DIR),
            filename  = f"{run_name}_{{epoch:02d}}_{{val_loss:.4f}}",
            monitor   = "val_loss",
            mode      = "min",
            save_top_k = 1,
            verbose   = True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    logger = CSVLogger(save_dir=str(LOG_DIR), name=run_name)

    trainer = pl.Trainer(
        max_epochs          = MAX_EPOCHS,
        accelerator         = "auto",
        gradient_clip_val   = GRADIENT_CLIP,
        callbacks           = callbacks,
        logger              = logger,
        enable_progress_bar = True,
        log_every_n_steps   = 10,
    )
    return trainer
