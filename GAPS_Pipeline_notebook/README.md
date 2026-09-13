# GAPS ETF 파이프라인

## 폴더 구조
```
GAPS_Pipeline/
├── run_weekly.py          ← ★ 매주 여기만 실행
├── config.py              ← ★ 키워드·API키 설정
├── requirements.txt
│
├── collectors/
│   ├── collect_macro.py   Step 1: 가격 + VIX/환율/금리차
│   ├── crawl_mk.py        Step 2a: 매일경제 크롤링
│   └── crawl_nyt.py       Step 2b: NYT 크롤링
│
├── processors/
│   ├── sentiment_score.py Step 3: 감성 점수 (KR-FinBERT + FinBERT)
│   └── build_dataset.py   Step 4: final_dataset.csv append
│
├── src_tft/               ← TFT 학습/예측 (기존 코드)
│   ├── config.py          DB GAPS 제약 + 188개 TICKER_ASSET_MAP
│   ├── train.py           Step 6: 모델 학습
│   ├── predict.py         Step 7: 예측 + 포트폴리오
│   ├── data/preprocess.py Step 5: 전처리
│   └── outputs/portfolio/ ← 최종 결과물
│
├── dashboard/
│   └── app.py             Streamlit 대시보드
│
└── data/
    ├── final_dataset.csv  ← 여기에 완성된 데이터 넣기
    ├── news/raw/
    ├── news/merged/
    └── outputs/logs/
```

## 설치
```bash
pip install -r requirements.txt
```

## 초기 설정 (config.py)
```python
BOK_API_KEY = "발급받은키"    # ecos.bok.or.kr/api (국고채 10년)
NYT_API_KEY = "발급받은키"    # developer.nytimes.com (선택)
```

## 데이터 준비
```
data/final_dataset.csv  ← 완성된 데이터셋 넣기
```

## 실행

### 매주 월요일
```bash
python run_weekly.py
```

### 처음 실행 (뉴스 없이 빠르게 예측만)
```bash
python run_weekly.py --skip-news --skip-price
```

### 모델 재학습 포함
```bash
python run_weekly.py --retrain
```

### 단계별
```bash
python run_weekly.py --step price      # 가격+거시만
python run_weekly.py --step news       # 뉴스만 (Chrome 창 뜸)
python run_weekly.py --step sentiment  # 감성만
python run_weekly.py --step dataset    # 데이터셋 갱신만
python run_weekly.py --step tft        # 전처리+예측만
python run_weekly.py --step train      # 학습만
python run_weekly.py --step predict    # 예측만
```

### 대시보드
```bash
streamlit run dashboard/app.py
# → http://localhost:8501
```

## 결과물
```
src_tft/outputs/portfolio/YYYY-MM-DD/
  ├── portfolio_conservative.csv
  ├── portfolio_aggressive.csv
  └── report.txt
```
