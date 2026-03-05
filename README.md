# Horse Race Prediction

A Python + Streamlit app that scrapes today's UK racecards from RacingPost,
computes scoring flags, and applies an XGBoost + LightGBM ensemble to predict
win probabilities.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Run the app
```bash
streamlit run app.py
```

### Train models (requires historical data)
Place accumulated `raceid.csv` and `results.csv` in `data/historical/`, then:
```bash
python src/train.py
```

### Scrape today's races (standalone)
```bash
python -c "from src.scraper import scrape_all_races_today; import json; df = scrape_all_races_today(); print(df.head())"
```

## Project Structure

- `app.py` — Streamlit dashboard (5 pages)
- `src/scraper.py` — Today's racecard scraper (RacingPost)
- `src/results_scraper.py` — Yesterday's results scraper
- `src/flags.py` — Flag1–Flag5 scoring system (preserved from original R logic)
- `src/features.py` — Feature engineering pipeline
- `src/train.py` — XGBoost + LightGBM training + calibration
- `src/predict.py` — Inference pipeline
- `data/Coursecountry.csv` — Racecourse → Country lookup
- `data/historical/` — Historical racecards + results for training
- `models/` — Saved model artifacts

## Scoring Flags (Rule-Based)

The Flag system provides quick rule-based picks without needing trained models:

- **Flag1** = TS + RPR − OR  (form vs official rating)
- **Flag2** = TS − weight  (topspeed adjusted for weight)
- **Flag3** = RPR − weight  (RP rating adjusted for weight)
- **Flag4** = (Flag1×2) + (Flag2×1) + (Flag3×1.75)  ← composite (primary)
- **Flag5** = (Flag1+Flag2+Flag3) / 3  ← simple average

## ML Models

- **XGBoost** + isotonic calibration
- **LightGBM** + isotonic calibration
- **Ensemble**: 50% XGB + 50% LGBM calibrated probabilities
- Win probabilities are normalized per race to sum to 1.0
