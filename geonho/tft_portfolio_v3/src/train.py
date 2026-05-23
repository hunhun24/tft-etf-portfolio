"""
train.py — 학습 진입점
실행: python src/train.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.preprocess import main as run_preprocess
from data.dataset    import load_preprocessed, build_datasets, build_dataloaders
from model.tft       import build_model, build_trainer
from config          import CHECKPOINT_DIR, PREPROCESSED_CSV


def main():
    # ── 1. 전처리 (outputs/data/final_dataset_preprocessed.csv 없으면 실행) ──
    if not PREPROCESSED_CSV.exists():
        print("=== Step 1: Preprocessing ===")
        run_preprocess()
    else:
        print(f"[skip] Preprocessed file already exists: {PREPROCESSED_CSV}")

    # ── 2. 데이터셋 / DataLoader 생성 ──
    print("\n=== Step 2: Building Datasets ===")
    df = load_preprocessed()
    train_ds, val_ds = build_datasets(df)
    train_loader, val_loader = build_dataloaders(train_ds, val_ds)
    print(f"  Train: {len(train_ds):,}  |  Val: {len(val_ds):,}")

    # ── 3. 모델 & Trainer 생성 ──
    print("\n=== Step 3: Building Model ===")
    model   = build_model(train_ds)
    trainer = build_trainer(run_name="tft_v1")

    # ── 4. 학습 ──
    print("\n=== Step 4: Training ===")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # ── 5. 결과 출력 ──
    best_ckpt = trainer.checkpoint_callback.best_model_path
    best_score = trainer.checkpoint_callback.best_model_score
    print(f"\n{'='*50}")
    print(f"학습 완료!")
    print(f"Best checkpoint : {best_ckpt}")
    print(f"Best val_loss   : {best_score:.6f}")
    print(f"{'='*50}")
    return best_ckpt


if __name__ == "__main__":
    main()
