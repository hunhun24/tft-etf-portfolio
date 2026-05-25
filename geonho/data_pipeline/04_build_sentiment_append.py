"""
Build MK/NYT sentiment append files.

기존 geonho/sentiment 파일과 news_clean_ver 산출물은 수정하지 않고,
geonho/data_pipeline/outputs 아래의 append 뉴스 파일에 대해서만 감성분석과
sentiment_score 계산을 수행한다.

Examples:
python geonho/data_pipeline/04_build_sentiment_append.py --limit 50
python data_pipeline/04_build_sentiment_append.py
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from torch.nn.functional import softmax
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


GEONHO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_OUTPUT_DIR = GEONHO_ROOT / "data_pipeline" / "outputs"
SENTIMENT_OUTPUT_DIR = PIPELINE_OUTPUT_DIR / "sentiment"

MK_INPUT_PATH = PIPELINE_OUTPUT_DIR / "mk" / "mk_news_geonho_2026_append_filtered.csv"
NYT_INPUT_PATH = PIPELINE_OUTPUT_DIR / "nyt" / "nyt_news_geonho_2026_append_clean.csv"

MK_SENTIMENT_PATH = SENTIMENT_OUTPUT_DIR / "mk_news_geonho_2026_append_sentiment.csv"
NYT_SENTIMENT_PATH = SENTIMENT_OUTPUT_DIR / "nyt_news_geonho_2026_append_sentiment.csv"

SCORED_FILES = {
    "mk": MK_SENTIMENT_PATH,
    "nyt": NYT_SENTIMENT_PATH,
}

SCORED_OUTPUTS = {
    "mk": SENTIMENT_OUTPUT_DIR / "mk_sentiment_scored_2026_append.csv",
    "nyt": SENTIMENT_OUTPUT_DIR / "nyt_sentiment_scored_2026_append.csv",
}

REQUIRED_INPUT_COLUMNS = ["keyword", "title", "date", "section"]


@dataclass(frozen=True)
class SentimentSpec:
    name: str
    input_path: Path
    sentiment_path: Path
    model_name: str
    label_order: tuple[str, str, str]


SPECS = [
    SentimentSpec(
        name="mk",
        input_path=MK_INPUT_PATH,
        sentiment_path=MK_SENTIMENT_PATH,
        model_name="snunlp/KR-FinBert-SC",
        label_order=("neg", "neu", "pos"),
    ),
    SentimentSpec(
        name="nyt",
        input_path=NYT_INPUT_PATH,
        sentiment_path=NYT_SENTIMENT_PATH,
        model_name="ProsusAI/finbert",
        label_order=("pos", "neg", "neu"),
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MK/NYT append sentiment inference and sentiment_score calculation."
    )
    parser.add_argument("--limit", type=int, default=None, help="Process only first n rows.")
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def validate_input(df: pd.DataFrame, path: Path) -> None:
    missing = [col for col in REQUIRED_INPUT_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")


def load_input(path: Path, limit: int | None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"input file not found: {path}")

    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [str(col).strip() for col in df.columns]
    validate_input(df, path)
    df = df[REQUIRED_INPUT_COLUMNS].copy()
    if limit is not None:
        df = df.head(limit).copy()
    return df.reset_index(drop=True)


def load_model(model_name: str, device: torch.device):
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return tokenizer, model


def predict_batch(
    texts: list[str],
    tokenizer,
    model,
    device: torch.device,
    label_order: tuple[str, str, str],
) -> list[dict[str, float]]:
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        logits = model(**encoded).logits
    probs = softmax(logits, dim=-1).cpu().numpy()

    results = []
    for prob in probs:
        row = {
            label_order[i]: round(float(prob[i]), 6)
            for i in range(len(label_order))
        }
        results.append({
            "pos": row["pos"],
            "neu": row["neu"],
            "neg": row["neg"],
        })
    return results


def run_sentiment(spec: SentimentSpec, limit: int | None, batch_size: int, device: torch.device) -> pd.DataFrame:
    print("\n" + "=" * 80)
    print(f"[{spec.name.upper()}] sentiment inference")
    print("=" * 80)
    df = load_input(spec.input_path, limit=limit)
    print(f"input: {spec.input_path}")
    print(f"input rows: {len(df):,}")

    tokenizer, model = load_model(spec.model_name, device)
    titles = df["title"].fillna("").astype(str).tolist()

    all_results = []
    for i in tqdm(range(0, len(titles), batch_size), desc=f"{spec.name} sentiment"):
        batch = titles[i:i + batch_size]
        all_results.extend(
            predict_batch(
                texts=batch,
                tokenizer=tokenizer,
                model=model,
                device=device,
                label_order=spec.label_order,
            )
        )

    df["pos"] = [row["pos"] for row in all_results]
    df["neu"] = [row["neu"] for row in all_results]
    df["neg"] = [row["neg"] for row in all_results]

    df.to_csv(spec.sentiment_path, index=False, encoding="utf-8-sig")
    print(f"sentiment output rows: {len(df):,}")
    print(f"saved: {spec.sentiment_path}")
    return df


def build_sentiment_scores() -> dict[str, pd.DataFrame]:
    """
    Existing sentiment_score.py logic with append paths:
    df["sentiment_score"] = (df["pos"] - df["neg"]).round(6)
    """
    scored = {}
    for name, path in SCORED_FILES.items():
        df = pd.read_csv(path, encoding="utf-8-sig")
        df["sentiment_score"] = (df["pos"] - df["neg"]).round(6)

        out_path = SCORED_OUTPUTS[name]
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        scored[name] = df

        print("\n" + "-" * 80)
        print(f"[{name.upper()}] scored output")
        print(f"input rows : {len(df):,}")
        print(f"saved      : {out_path}")
        print("sentiment_score describe:")
        print(df["sentiment_score"].describe().round(4).to_string())

    return scored


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError(f"batch-size must be positive, got {args.batch_size}")
    if args.limit is not None and args.limit <= 0:
        raise ValueError(f"limit must be positive when set, got {args.limit}")

    SENTIMENT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"limit: {args.limit}")
    print(f"batch_size: {args.batch_size}")
    print(f"output dir: {SENTIMENT_OUTPUT_DIR}")

    sentiment_outputs = {}
    for spec in SPECS:
        sentiment_outputs[spec.name] = run_sentiment(
            spec=spec,
            limit=args.limit,
            batch_size=args.batch_size,
            device=device,
        )

    scored = build_sentiment_scores()

    print("\n" + "=" * 80)
    print("Sentiment append summary")
    print("=" * 80)
    for spec in SPECS:
        print(f"[{spec.name.upper()}]")
        print(f"  input rows            : {len(sentiment_outputs[spec.name]):,}")
        print(f"  sentiment output rows : {len(sentiment_outputs[spec.name]):,}")
        print(f"  scored output rows    : {len(scored[spec.name]):,}")
        print(f"  sentiment file        : {spec.sentiment_path}")
        print(f"  scored file           : {SCORED_OUTPUTS[spec.name]}")


if __name__ == "__main__":
    main()
