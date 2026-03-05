"""
Inference pipeline: load trained models, score horses, ensemble + per-race normalization.
"""

import json
import logging
import os
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb

logger = logging.getLogger(__name__)

MODELS_DIR = "models"
XGB_MODEL_PATH = os.path.join(MODELS_DIR, "xgb_model.json")
LGBM_MODEL_PATH = os.path.join(MODELS_DIR, "lgbm_model.txt")
CAL_XGB_PATH = os.path.join(MODELS_DIR, "calibrator_xgb.pkl")
CAL_LGBM_PATH = os.path.join(MODELS_DIR, "calibrator_lgbm.pkl")
PREPROCESSOR_PATH = os.path.join(MODELS_DIR, "preprocessor.pkl")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")

# Probability thresholds (from user's original R script: 0.39 for CV, 0.44 for RF)
# We apply a single ensemble threshold for selections
ENSEMBLE_THRESHOLD = 0.40


def models_available() -> bool:
    """Return True if all required model artifacts exist."""
    return all(
        os.path.exists(p)
        for p in [XGB_MODEL_PATH, LGBM_MODEL_PATH, CAL_XGB_PATH, CAL_LGBM_PATH, PREPROCESSOR_PATH]
    )


def load_models() -> dict:
    """Load all model artifacts and return as a dict."""
    if not models_available():
        raise FileNotFoundError(
            "Model files not found. Run training first via the 'Train / Evaluate' tab "
            "or: python src/train.py"
        )

    xgb_model = xgb.Booster()
    xgb_model.load_model(XGB_MODEL_PATH)

    lgbm_model = lgb.Booster(model_file=LGBM_MODEL_PATH)

    cal_xgb = joblib.load(CAL_XGB_PATH)
    cal_lgbm = joblib.load(CAL_LGBM_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    logger.info("All model artifacts loaded successfully.")
    return {
        "xgb": xgb_model,
        "lgbm": lgbm_model,
        "cal_xgb": cal_xgb,
        "cal_lgbm": cal_lgbm,
        "preprocessor": preprocessor,
    }


def predict_win_probabilities(
    df: pd.DataFrame,
    models: dict,
    normalize_per_race: bool = True,
    xgb_weight: float = 0.5,
    lgbm_weight: float = 0.5,
) -> pd.DataFrame:
    """
    Score horses and return win probabilities.

    Args:
        df: prepared feature DataFrame (output of features.prepare_features)
        models: dict from load_models()
        normalize_per_race: if True, probs are normalized to sum to 1.0 per race
        xgb_weight: weight for XGBoost in ensemble (default 0.5)
        lgbm_weight: weight for LightGBM in ensemble (default 0.5)

    Returns:
        Input DataFrame with added columns:
            prob_xgb, prob_lgbm, prob_ensemble, win_probability,
            implied_odds, is_selection (bool, ensemble >= threshold)
    """
    from src.features import transform, ALL_FEATURES

    if df.empty:
        return df

    # Keep non-feature columns for output
    id_cols = [c for c in ["race_id", "race_time", "race_date", "racecourse", "horse_name",
                            "horse_no", "Flag1", "Flag2", "Flag3", "Flag4", "Flag5",
                            "OR", "TS", "RPR", "going", "draw", "won"]
               if c in df.columns]

    X = transform(df, models["preprocessor"])

    # XGBoost
    dmatrix = xgb.DMatrix(X)
    raw_xgb = models["xgb"].predict(dmatrix)
    prob_xgb = models["cal_xgb"].predict(raw_xgb)

    # LightGBM
    raw_lgbm = models["lgbm"].predict(X)
    prob_lgbm = models["cal_lgbm"].predict(raw_lgbm)

    # Ensemble
    assert abs(xgb_weight + lgbm_weight - 1.0) < 1e-6, "Weights must sum to 1"
    prob_ensemble = xgb_weight * prob_xgb + lgbm_weight * prob_lgbm

    result = df[id_cols].copy()
    result["prob_xgb"] = np.clip(prob_xgb, 0, 1)
    result["prob_lgbm"] = np.clip(prob_lgbm, 0, 1)
    result["prob_ensemble"] = np.clip(prob_ensemble, 0, 1)

    if normalize_per_race:
        # Per-race normalization so probabilities sum to 1.0 per race
        race_sums = result.groupby("race_id")["prob_ensemble"].transform("sum")
        result["win_probability"] = result["prob_ensemble"] / race_sums.clip(lower=1e-8)
    else:
        result["win_probability"] = result["prob_ensemble"]

    # Implied odds (1/probability)
    result["implied_odds"] = (1.0 / result["win_probability"].clip(lower=0.01)).round(1)

    # Selection flag
    result["is_selection"] = result["prob_ensemble"] >= ENSEMBLE_THRESHOLD

    # Sort: race_time → race_id → win_probability desc
    result = result.sort_values(
        ["race_time", "race_id", "win_probability"],
        ascending=[True, True, False],
    ).reset_index(drop=True)

    return result


def get_top_ml_picks(
    predictions: pd.DataFrame,
    n: int = 5,
    selections_only: bool = False,
) -> pd.DataFrame:
    """
    Return top n horses per race by win_probability.

    Args:
        predictions: output of predict_win_probabilities
        n: number of top horses per race
        selections_only: if True, only return horses above the ensemble threshold
    """
    df = predictions.copy()
    if selections_only:
        df = df[df["is_selection"]]

    df["ml_rank"] = df.groupby("race_id")["win_probability"].rank(
        ascending=False, method="first"
    ).astype(int)

    top = df[df["ml_rank"] <= n].copy()
    top = top.sort_values(["race_time", "race_id", "ml_rank"]).reset_index(drop=True)
    return top


def load_metrics() -> Optional[dict]:
    """Load saved evaluation metrics if available."""
    if not os.path.exists(METRICS_PATH):
        return None
    with open(METRICS_PATH) as f:
        return json.load(f)


def join_predictions_with_results(
    predictions: pd.DataFrame,
    results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join yesterday's predictions against actual results.

    Args:
        predictions: saved predictions DataFrame (with race_id, horse_name, win_probability, is_selection)
        results: scraped results DataFrame (with race_id, horse_name, result_pos)

    Returns:
        Merged DataFrame with: actual_position, actual_winner (bool), in_top3 (bool)
    """
    pred = predictions.copy()
    res = results.copy()

    # Normalize for join
    pred["horse_name_norm"] = pred["horse_name"].str.upper().str.strip()
    res["horse_name_norm"] = res["horse_name"].str.upper().str.strip()

    merged = pred.merge(
        res[["race_id", "horse_name_norm", "result_pos"]],
        on=["race_id", "horse_name_norm"],
        how="left",
    )

    merged["result_pos"] = pd.to_numeric(merged["result_pos"], errors="coerce")
    merged["actual_winner"] = merged["result_pos"] == 1
    merged["in_top3"] = merged["result_pos"].between(1, 3, inclusive="both")

    return merged


def compute_hit_rates(merged: pd.DataFrame) -> dict:
    """
    Compute prediction hit rates.

    Returns:
        dict with keys: flag4_win_rate, flag4_top3_rate, ml_win_rate, ml_top3_rate
    """
    # Flag4 top pick (flag_rank==1 if present, else flag4_rank_in_race==1)
    flag_col = "flag_rank" if "flag_rank" in merged.columns else None
    ml_col = "ml_rank" if "ml_rank" in merged.columns else None

    out = {}

    if flag_col and flag_col in merged.columns:
        flag_top = merged[merged[flag_col] == 1]
        out["flag4_win_rate"] = float(flag_top["actual_winner"].mean()) if not flag_top.empty else None
        out["flag4_top3_rate"] = float(flag_top["in_top3"].mean()) if not flag_top.empty else None

    if ml_col and ml_col in merged.columns:
        ml_top = merged[merged[ml_col] == 1]
        out["ml_win_rate"] = float(ml_top["actual_winner"].mean()) if not ml_top.empty else None
        out["ml_top3_rate"] = float(ml_top["in_top3"].mean()) if not ml_top.empty else None

    return out
