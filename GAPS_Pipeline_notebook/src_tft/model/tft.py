"""
tft.py
------
TemporalFusionTransformer 인스턴스화 및 PyTorch Lightning Trainer 설정.
"""

import os
import sys

import lightning as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import MAE, RMSE, QuantileLoss
from lightning.pytorch.loggers import CSVLogger


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import (
    MAX_ENCODER_LENGTH,
    MAX_PREDICTION_LENGTH,
    CHECKPOINT_DIR,
    LOG_DIR,
    FEATURE_VERSION,
    HIDDEN_SIZE,
    ATTENTION_HEAD_SIZE,
    DROPOUT,
    HIDDEN_CONTINUOUS_SIZE,
    LEARNING_RATE,
    MAX_EPOCHS,
    PATIENCE,
)

QUANTILES = [0.1, 0.5, 0.9]   # 불확실성 구간 + 중앙값(point prediction)

def build_model(train_ds) -> TemporalFusionTransformer:
    """
    TimeSeriesDataSet으로부터 TFT 모델 생성.

    Parameters
    ----------
    train_ds : TimeSeriesDataSet
        dataset.py의 make_datasets()에서 반환된 train dataset.
        피처 구조(categorical embedding 크기 등)를 자동으로 추론.

    Returns
    -------
    TemporalFusionTransformer
    """
    model = TemporalFusionTransformer.from_dataset(
        train_ds,

        # ── 아키텍처 ──────────────────────────────────────────────────────────
        hidden_size=HIDDEN_SIZE,
        attention_head_size=ATTENTION_HEAD_SIZE,
        dropout=DROPOUT,
        hidden_continuous_size=HIDDEN_CONTINUOUS_SIZE,

        # ── Loss / Metrics ────────────────────────────────────────────────────
        loss=QuantileLoss(quantiles=QUANTILES),   # 학습: 분위수 손실
        logging_metrics=[MAE(), RMSE()],           # val 로깅: 베이스라인 비교용

        # ── 옵티마이저 ────────────────────────────────────────────────────────
        learning_rate=LEARNING_RATE,
        optimizer="adam",

        # ── 로깅 ──────────────────────────────────────────────────────────────
        log_interval=10,        # 10 step마다 loss 로깅
        log_val_interval=1,     # val epoch마다 로깅
    )

    return model


def build_trainer(
    max_epochs: int = MAX_EPOCHS,
    logger_name: str = None,
    gradient_clip_val: float = 0.1,
    min_epochs: int = 10,
) -> pl.Trainer:
    ckpt_subdir  = logger_name if logger_name is not None else FEATURE_VERSION
    _logger_name = logger_name if logger_name is not None else FEATURE_VERSION

    early_stop = EarlyStopping(
        monitor="val_loss",
        min_delta=1e-5,
        patience=PATIENCE,
        mode="min",
        verbose=True,
    )

    checkpoint = ModelCheckpoint(
        dirpath=os.path.join(CHECKPOINT_DIR, ckpt_subdir),
        filename=f"tft-{ckpt_subdir}-{{epoch:02d}}-{{val_loss:.4f}}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )

    logger = CSVLogger(save_dir=LOG_DIR, name=_logger_name)

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        min_epochs=min_epochs,
        accelerator="auto",
        gradient_clip_val=gradient_clip_val,
        callbacks=[early_stop, checkpoint],
        logger=logger,
        enable_progress_bar=True,
        deterministic=False,
    )

    return trainer


# ── sanity check ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from data.dataset import make_datasets

    train_ds, val_ds, train_loader, val_loader = make_datasets()

    model = build_model(train_ds)
    print(model)
    print(f"\n학습 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")