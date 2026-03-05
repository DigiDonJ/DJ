"""
Model training pipeline: XGBoost + LightGBM + isotonic calibration.

Usage (standalone):
    python src/train.py

Requires:
    data/historical/raceid.csv   — historical racecards
    data/historical/results.csv  — historical results (with result_pos column)
"""

import json
import logging
import os
import sys

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss

# Allow running as a script from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features import (
    prepare_features,
    fit_preprocessor,
    transform,
    ALL_FEATURES,
    TARGET_COLUMN,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = "models"
XGB_MODEL_PATH = os.path.join(MODELS_DIR, "xgb_model.json")
LGBM_MODEL_PATH = os.path.join(MODELS_DIR, "lgbm_model.txt")
CAL_XGB_PATH = os.path.join(MODELS_DIR, "calibrator_xgb.pkl")
CAL_LGBM_PATH = os.path.join(MODELS_DIR, "calibrator_lgbm.pkl")
PREPROCESSOR_PATH = os.path.join(MODELS_DIR, "preprocessor.pkl")
FEATURE_COLS_PATH = os.path.join(MODELS_DIR, "feature_columns.json")


# ---------------------------------------------------------------------------
# Data loading & labelling
# ---------------------------------------------------------------------------

def load_training_data(
    raceid_path: str = "data/historical/raceid.csv",
    results_path: str = "data/historical/results.csv",
    coursecountry_path: str = "data/Coursecountry.csv",
) -> pd.DataFrame:
    """
    Load historical racecards + results, join them, and label winners (won=1).

    The join key is race_id + horse_no/horse_num.
    """
    if not os.path.exists(raceid_path):
        raise FileNotFoundError(f"Historical racecard file not found: {raceid_path}")
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Historical results file not found: {results_path}")

    cards = pd.read_csv(raceid_path, low_memory=False)
    results = pd.read_csv(results_path, low_memory=False)

    # Normalise column names
    cards.columns = cards.columns.str.strip().str.lower().str.replace(" ", "_")
    results.columns = results.columns.str.strip().str.lower().str.replace(" ", "_")

    # Ensure consistent key names
    if "horse_no" not in cards.columns and "horseno" in cards.columns:
        cards = cards.rename(columns={"horseno": "horse_no"})
    if "horse_num" not in results.columns and "resulthorsenum" in results.columns:
        results = results.rename(columns={"resulthorsenum": "horse_num"})
    if "race_id" not in results.columns and "resultraceid" in results.columns:
        results = results.rename(columns={"resultraceid": "race_id"})
    if "result_pos" not in results.columns and "result_pos" in results.columns:
        pass  # already named

    # Keep only winner rows for labelling (result_pos == 1)
    winners = results[pd.to_numeric(results.get("result_pos"), errors="coerce") == 1][
        ["race_id", "horse_num"]
    ].copy()
    winners["race_id"] = winners["race_id"].astype(str)
    winners["horse_num"] = pd.to_numeric(winners["horse_num"], errors="coerce")

    cards["race_id"] = cards["race_id"].astype(str)
    cards["horse_no"] = pd.to_numeric(cards.get("horse_no"), errors="coerce")

    # Label: won = 1 if horse is the winner in that race
    cards = cards.merge(
        winners.rename(columns={"horse_num": "horse_no"}).assign(won=1),
        on=["race_id", "horse_no"],
        how="left",
    )
    cards["won"] = cards["won"].fillna(0).astype(int)

    # Only keep races where exactly one winner is labeled (data quality)
    winner_counts = cards.groupby("race_id")["won"].sum()
    valid_races = winner_counts[winner_counts == 1].index
    cards = cards[cards["race_id"].isin(valid_races)].copy()

    logger.info(
        f"Loaded {len(cards)} horse entries across {cards['race_id'].nunique()} races "
        f"({cards['won'].sum()} winners labeled)"
    )

    # Run feature engineering
    df = prepare_features(cards, coursecountry_path=coursecountry_path, england_only=True)
    return df


# ---------------------------------------------------------------------------
# Train/val time-based split
# ---------------------------------------------------------------------------

def time_split(df: pd.DataFrame, val_fraction: float = 0.2):
    """Split by race_date: last val_fraction of dates goes to validation."""
    dates = pd.to_datetime(df["race_date"], errors="coerce")
    df = df.copy()
    df["_date"] = dates
    sorted_dates = df["_date"].dropna().sort_values().unique()
    cutoff_idx = int(len(sorted_dates) * (1 - val_fraction))
    cutoff = sorted_dates[cutoff_idx]
    train = df[df["_date"] < cutoff].drop(columns=["_date"])
    val = df[df["_date"] >= cutoff].drop(columns=["_date"])
    logger.info(f"Train: {len(train)} rows | Val: {len(val)} rows (cutoff: {cutoff.date()})")
    return train, val


# ---------------------------------------------------------------------------
# XGBoost training
# ---------------------------------------------------------------------------

def train_xgboost(X_train, y_train, X_val, y_val) -> xgb.Booster:
    params = {
        "objective": "binary:logistic",
        "eta": 0.05,
        "max_depth": 5,
        "min_child_weight": 10,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "lambda": 2.0,
        "alpha": 0.5,
        "eval_metric": "logloss",
        "seed": 42,
    }

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    evals_result = {}
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=2000,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=30,
        evals_result=evals_result,
        verbose_eval=100,
    )

    best_round = model.best_iteration
    logger.info(f"XGBoost best iteration: {best_round}")
    return model


# ---------------------------------------------------------------------------
# LightGBM training
# ---------------------------------------------------------------------------

def train_lightgbm(X_train, y_train, X_val, y_val) -> lgb.Booster:
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "reg_lambda": 2.0,
        "reg_alpha": 0.5,
        "seed": 42,
        "verbosity": -1,
    }

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    callbacks = [lgb.early_stopping(30, verbose=False), lgb.log_evaluation(100)]

    model = lgb.train(
        params,
        train_data,
        num_boost_round=2000,
        valid_sets=[train_data, val_data],
        valid_names=["train", "val"],
        callbacks=callbacks,
    )

    logger.info(f"LightGBM best iteration: {model.best_iteration}")
    return model


# ---------------------------------------------------------------------------
# Isotonic calibration
# ---------------------------------------------------------------------------

def fit_calibrator(raw_probs: np.ndarray, y_true: np.ndarray) -> IsotonicRegression:
    cal = IsotonicRegression(out_of_bounds="clip")
    cal.fit(raw_probs, y_true)
    return cal


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def rank_accuracy(df_val: pd.DataFrame, prob_col: str) -> float:
    """
    Rank accuracy: fraction of races where the highest-probability horse won.
    """
    df_val = df_val.copy()
    df_val["_prob"] = df_val[prob_col]
    top_pred = df_val.loc[df_val.groupby("race_id")["_prob"].idxmax()]
    return float((top_pred["won"] == 1).mean())


def evaluate(
    df_val: pd.DataFrame,
    y_val: np.ndarray,
    probs_xgb: np.ndarray,
    probs_lgbm: np.ndarray,
) -> dict:
    ensemble = (probs_xgb + probs_lgbm) / 2.0

    df_val = df_val.copy()
    df_val["prob_xgb"] = probs_xgb
    df_val["prob_lgbm"] = probs_lgbm
    df_val["prob_ensemble"] = ensemble

    results = {}
    for name, probs in [("xgb", probs_xgb), ("lgbm", probs_lgbm), ("ensemble", ensemble)]:
        ll = log_loss(y_val, probs)
        auc = roc_auc_score(y_val, probs)
        bs = brier_score_loss(y_val, probs)
        ra = rank_accuracy(df_val, f"prob_{name}")
        results[name] = {"log_loss": ll, "roc_auc": auc, "brier_score": bs, "rank_accuracy": ra}
        logger.info(
            f"{name.upper():10s} | LogLoss={ll:.4f} | AUC={auc:.4f} | "
            f"Brier={bs:.4f} | RankAcc={ra:.3f}"
        )

    return results


# ---------------------------------------------------------------------------
# Main training orchestration
# ---------------------------------------------------------------------------

def train(
    raceid_path: str = "data/historical/raceid.csv",
    results_path: str = "data/historical/results.csv",
    coursecountry_path: str = "data/Coursecountry.csv",
) -> dict:
    """
    Full training pipeline. Returns evaluation metrics.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    # 1. Load and label data
    df = load_training_data(raceid_path, results_path, coursecountry_path)

    if len(df) < 200:
        raise ValueError(
            f"Insufficient training data: only {len(df)} rows after filtering. "
            "Need more historical data in data/historical/."
        )

    # 2. Time-based split
    train_df, val_df = time_split(df)

    # 3. Fit preprocessor on training data only
    preprocessor = fit_preprocessor(train_df, save_path=PREPROCESSOR_PATH)

    X_train = transform(train_df, preprocessor)
    y_train = train_df[TARGET_COLUMN].values

    X_val = transform(val_df, preprocessor)
    y_val = val_df[TARGET_COLUMN].values

    # 4. Train models
    logger.info("Training XGBoost...")
    xgb_model = train_xgboost(X_train, y_train, X_val, y_val)
    xgb_model.save_model(XGB_MODEL_PATH)
    logger.info(f"XGBoost saved to {XGB_MODEL_PATH}")

    logger.info("Training LightGBM...")
    lgbm_model = train_lightgbm(X_train, y_train, X_val, y_val)
    lgbm_model.save_model(LGBM_MODEL_PATH)
    logger.info(f"LightGBM saved to {LGBM_MODEL_PATH}")

    # 5. Raw validation probabilities
    dval = xgb.DMatrix(X_val)
    raw_xgb = xgb_model.predict(dval)
    raw_lgbm = lgbm_model.predict(X_val)

    # 6. Isotonic calibration
    logger.info("Fitting isotonic calibrators...")
    cal_xgb = fit_calibrator(raw_xgb, y_val)
    cal_lgbm = fit_calibrator(raw_lgbm, y_val)
    joblib.dump(cal_xgb, CAL_XGB_PATH)
    joblib.dump(cal_lgbm, CAL_LGBM_PATH)

    # 7. Calibrated probs
    probs_xgb = cal_xgb.predict(raw_xgb)
    probs_lgbm = cal_lgbm.predict(raw_lgbm)

    # 8. Evaluate
    logger.info("Evaluating models...")
    metrics = evaluate(val_df, y_val, probs_xgb, probs_lgbm)

    # 9. Save metrics
    metrics_path = os.path.join(MODELS_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")

    return metrics


if __name__ == "__main__":
    train()
