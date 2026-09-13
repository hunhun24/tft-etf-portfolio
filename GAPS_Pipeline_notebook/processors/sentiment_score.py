"""
processors/sentiment_score.py
KR-FinBERT (매일경제) + FinBERT (NYT) 감성 점수 산출
"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import torch
from torch.nn.functional import softmax
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config import RAW_DIR, MERGED_DIR

BATCH_SIZE = 32
device = "cuda" if torch.cuda.is_available() else "cpu"

_models = {}

def _load_model(name: str):
    if name not in _models:
        print(f"  모델 로딩: {name} ({device})")
        tokenizer = AutoTokenizer.from_pretrained(name)
        model     = AutoModelForSequenceClassification.from_pretrained(name)
        model.eval().to(device)
        _models[name] = (tokenizer, model)
    return _models[name]


def _score_batch(texts: list, model_name: str, is_korean: bool) -> list:
    tokenizer, model = _load_model(model_name)
    all_scores = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch  = texts[i:i+BATCH_SIZE]
        inputs = tokenizer(batch, padding=True, truncation=True,
                           max_length=128, return_tensors="pt").to(device)
        with torch.no_grad():
            probs = softmax(model(**inputs).logits, dim=-1).cpu().numpy()

        if is_korean:
            # KR-FinBERT: {0: negative, 1: neutral, 2: positive}
            scores = probs[:, 2] - probs[:, 0]
        else:
            # FinBERT: {0: positive, 1: negative, 2: neutral}
            scores = probs[:, 0] - probs[:, 1]

        all_scores.extend(scores.tolist())
        if (i // BATCH_SIZE) % 10 == 0:
            print(f"    [{i}/{len(texts)}]", end="\r")
    print()
    return all_scores


def score_mk(sector: str) -> pd.DataFrame:
    """매일경제 감성 점수 (KR-FinBERT) → 날짜별 domestic 집계"""
    path = RAW_DIR / f"mk_{sector}.csv"
    if not path.exists():
        print(f"  [MK/{sector}] 파일 없음: {path}")
        return pd.DataFrame()

    df = pd.read_csv(path, encoding="utf-8-sig")
    df["date"] = pd.to_datetime(df.get("date", df.get("날짜","")), errors="coerce")
    df = df.dropna(subset=["date","제목"])
    df = df[df["제목"].str.strip() != ""]

    if df.empty:
        return pd.DataFrame()

    print(f"  [MK/{sector}] {len(df)}건 추론...")
    df["sentiment"] = _score_batch(df["제목"].tolist(), "snunlp/KR-FinBert-SC", is_korean=True)

    daily = (
        df.groupby(df["date"].dt.strftime("%Y-%m-%d"))["sentiment"]
        .agg(domestic_mean="mean",
             domestic_std=lambda x: x.std() if len(x)>1 else 0.0,
             domestic_count="count")
        .reset_index()
        .rename(columns={"date":"date"})
    )
    daily["sector"] = sector
    daily["domestic_std"] = daily["domestic_std"].fillna(0)
    return daily[["date","sector","domestic_mean","domestic_std","domestic_count"]]


def score_nyt(sector: str) -> pd.DataFrame:
    """NYT 감성 점수 (FinBERT) → 날짜별 global 집계"""
    path = RAW_DIR / f"nyt_{sector}.csv"
    if not path.exists():
        print(f"  [NYT/{sector}] 파일 없음: {path}")
        return pd.DataFrame()

    df = pd.read_csv(path, encoding="utf-8-sig")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date","title"])
    df = df[df["title"].str.strip() != ""]

    if df.empty:
        return pd.DataFrame()

    print(f"  [NYT/{sector}] {len(df)}건 추론...")
    df["sentiment"] = _score_batch(df["title"].tolist(), "ProsusAI/finbert", is_korean=False)

    daily = (
        df.groupby(df["date"].dt.strftime("%Y-%m-%d"))["sentiment"]
        .agg(global_mean="mean",
             global_std=lambda x: x.std() if len(x)>1 else 0.0,
             global_count="count")
        .reset_index()
        .rename(columns={"date":"date"})
    )
    daily["sector"] = sector
    daily["global_std"] = daily["global_std"].fillna(0)
    return daily[["date","sector","global_mean","global_std","global_count"]]


def score_all_sectors(sectors: list = None) -> pd.DataFrame:
    """전체 섹터 감성 점수 산출 → 날짜×sector 테이블"""
    from config import MK_KEYWORDS
    target = sectors or list(MK_KEYWORDS.keys())

    all_domestic, all_global = [], []
    for sector in target:
        d = score_mk(sector)
        g = score_nyt(sector)
        if not d.empty: all_domestic.append(d)
        if not g.empty: all_global.append(g)

    dom = pd.concat(all_domestic, ignore_index=True) if all_domestic else pd.DataFrame()
    glo = pd.concat(all_global,   ignore_index=True) if all_global   else pd.DataFrame()

    if dom.empty and glo.empty:
        return pd.DataFrame()

    # 날짜×sector 뼈대
    dates = pd.date_range("2026-01-01", pd.Timestamp.today(), freq="B").strftime("%Y-%m-%d")
    base  = pd.DataFrame([(d,s) for d in dates for s in target], columns=["date","sector"])

    sent_cols = ["domestic_mean","domestic_std","domestic_count",
                 "global_mean","global_std","global_count"]

    result = base.copy()
    if not dom.empty:
        result = result.merge(dom, on=["date","sector"], how="left")
    else:
        for c in ["domestic_mean","domestic_std","domestic_count"]:
            result[c] = 0.0

    if not glo.empty:
        result = result.merge(glo, on=["date","sector"], how="left")
    else:
        for c in ["global_mean","global_std","global_count"]:
            result[c] = 0.0

    result[sent_cols] = result[sent_cols].fillna(0)

    out = MERGED_DIR / "sentiment_daily.csv"
    result.to_csv(out, index=False)
    print(f"  ✅ 감성 저장: {out} ({len(result)}행)")
    return result
