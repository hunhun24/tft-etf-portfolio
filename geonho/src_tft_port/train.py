"""
train.py
--------
TFT 학습 진입점.

실행:
    python src/train.py
"""

import os
import sys

sys.path.append(os.path.dirname(__file__))

from data.dataset import make_datasets
from model.tft import build_model, build_trainer
from config import MAX_EPOCHS, gradient_clip_val


def main():
    # 1. 데이터셋 / DataLoader
    print("=" * 60)
    print("Loading datasets...")
    train_ds, val_ds, train_loader, val_loader = make_datasets()
    print(f"  Train samples : {len(train_ds)}")
    print(f"  Val   samples : {len(val_ds)}")

    # 2. 모델
    print("\nBuilding model...")
    model = build_model(train_ds)
    print(f"  Parameters    : {sum(p.numel() for p in model.parameters()):,}")

    # 3. Trainer
    print("\nBuilding trainer...")
    trainer = build_trainer(max_epochs=MAX_EPOCHS, gradient_clip_val=gradient_clip_val)

    # 4. 학습
    print("\nTraining...")
    print("=" * 60)
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    # 5. 결과
    print("=" * 60)
    print("Training complete.")
    best_ckpt = trainer.checkpoint_callback.best_model_path
    print(f"  Best checkpoint : {best_ckpt}")
    print(f"  Best val_loss   : {trainer.checkpoint_callback.best_model_score:.6f}")


if __name__ == "__main__":
    main()