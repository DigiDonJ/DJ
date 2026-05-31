"""
Feature engineering pipeline for the horse race prediction ML models.
"""

import json
import logging
import os
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature column definitions
# ---------------------------------------------------------------------------

NUMERIC_FEATURES = [
    # Ratings — kept as NaN when absent; models learn from missingness natively
    "OR", "TS", "RPR",
    # Flag scores (computed when ratings exist, NaN otherwise)
    "Flag1", "Flag2", "Flag3", "Flag4", "Flag5",
    # Horse / race attributes
    "draw", "last_run", "horse_age", "weight", "weight_relative",
    "price_money", "racewkday", "racemonth", "trainer_RFT",
    # Race-context features
    "field_size", "flag4_rank_in_race", "flag4_z_in_race", "or_pct_in_race",
    # Form-string derived (always computable)
    "form_win_rate", "form_place_rate", "form_recent_score",
    "form_run_count", "form_last1",
]

CATEGORICAL_FEATURES = [
    "going",
    "racecourse",
    "race_time",
]

BOOLEAN_FEATURES = [
    "missing_trainerRFT",
    "missing_draw",
    "missing_lastrun",
    # Ratings missingness — carry real signal (maiden, returning, unraced)
    "missing_OR",
    "missing_TS",
    "missing_RPR",
    # Form quality signal
    "form_had_dnf",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BOOLEAN_FEATURES
TARGET_COLUMN = "won"


# ---------------------------------------------------------------------------
# Form string parser
# ---------------------------------------------------------------------------

def _parse_form(form_str) -> dict:
    """
    Parse a RacingPost form string (e.g. '2131F20') into numeric features.

    Returns a dict with: win_rate, place_rate, recent_score, run_count,
                         last1 (most recent position), had_dnf.
    """
    null = {"form_win_rate": np.nan, "form_place_rate": np.nan,
            "form_recent_score": np.nan, "form_run_count": np.nan,
            "form_last1": np.nan, "form_had_dnf": False}

    if not form_str or pd.isna(form_str):
        return null

    s = str(form_str).strip().upper()
    # Take last 6 characters (most recent 6 runs)
    s = s[-6:]

    positions = []
    had_dnf = False
    for ch in s:
        if ch.isdigit():
            positions.append(int(ch))  # 0 = 10th or worse on RP
        elif ch in ("F", "U", "B", "P", "R", "S", "C", "D", "L"):
            positions.append(99)       # non-finisher / disqualified
            had_dnf = True
        # "/" separators and letters like "W" (won) are ignored

    if not positions:
        return null

    n = len(positions)
    finishers = [p for p in positions if p < 99]
    wins = sum(1 for p in positions if p == 1)
    places = sum(1 for p in positions if 1 <= p <= 3)

    # Exponentially weighted recent form score (latest run = highest weight)
    decay = 0.7
    weights = [decay ** i for i in range(n)][::-1]  # oldest first → latest last
    weighted_pos = [p if p < 99 else 10 for p in positions]  # cap non-finishers at 10
    recent_score = float(np.average(weighted_pos, weights=weights))

    return {
        "form_win_rate": wins / n,
        "form_place_rate": places / n,
        "form_recent_score": recent_score,
        "form_run_count": float(n),
        "form_last1": float(positions[-1]) if positions[-1] < 99 else 10.0,
        "form_had_dnf": had_dnf,
    }


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_features(
    df: pd.DataFrame,
    coursecountry_path: str = "data/Coursecountry.csv",
    england_only: bool = True,
) -> pd.DataFrame:
    df = df.copy()

    # --- Join with Coursecountry ---
    if os.path.exists(coursecountry_path):
        cc = pd.read_csv(coursecountry_path)
        cc.columns = cc.columns.str.strip()
        df = df.merge(cc, on="racecourse", how="left")
    else:
        logger.warning(f"Coursecountry file not found at {coursecountry_path}. Skipping country filter.")
        df["Country"] = "ENGLAND"

    if england_only:
        df = df[df["Country"].str.upper() == "ENGLAND"].copy()

    if df.empty:
        return df

    # --- Compute flags if missing (historical R CSVs have raw ratings only) ---
    if "Flag4" not in df.columns:
        or_ = pd.to_numeric(df.get("OR"), errors="coerce")
        ts  = pd.to_numeric(df.get("TS"), errors="coerce")
        rpr = pd.to_numeric(df.get("RPR"), errors="coerce")
        w   = pd.to_numeric(df.get("weight"), errors="coerce")
        df["Flag1"] = ts + rpr - or_
        df["Flag2"] = ts - w
        df["Flag3"] = rpr - w
        df["Flag4"] = (df["Flag1"] * 2) + (df["Flag2"] * 1) + (df["Flag3"] * 1.75)
        df["Flag5"] = (df["Flag1"] + df["Flag2"] + df["Flag3"]) / 3

    # --- Ratings missingness indicators (BEFORE any coercion) ---
    df["missing_OR"]  = pd.to_numeric(df.get("OR"),  errors="coerce").isna()
    df["missing_TS"]  = pd.to_numeric(df.get("TS"),  errors="coerce").isna()
    df["missing_RPR"] = pd.to_numeric(df.get("RPR"), errors="coerce").isna()

    # --- Date features ---
    dates = pd.to_datetime(df["race_date"], format="%Y-%m-%d", errors="coerce")
    df["racemonth"] = dates.dt.month
    df["racewkday"] = dates.dt.dayofweek

    # --- Standard missing indicators ---
    df["missing_trainerRFT"] = df.get("trainer_RFT", pd.Series(dtype=float)).isna()
    df["missing_draw"]       = df["draw"].fillna(0) == 0 if "draw" in df.columns else True
    df["missing_lastrun"]    = df.get("last_run", pd.Series(dtype=float)).isna()

    # --- Field size ---
    df["field_size"] = df.groupby("race_id")["race_id"].transform("count")

    # --- Race-relative features ---
    df["Flag4_num"] = pd.to_numeric(df["Flag4"], errors="coerce")
    df["flag4_rank_in_race"] = df.groupby("race_id")["Flag4_num"].rank(
        ascending=False, method="first", na_option="bottom"
    )
    df["flag4_z_in_race"] = df.groupby("race_id")["Flag4_num"].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-8)
    )
    or_num = pd.to_numeric(df.get("OR"), errors="coerce")
    if "OR" in df.columns:
        df["or_pct_in_race"] = or_num / df.groupby("race_id")["OR"].transform(
            lambda x: pd.to_numeric(x, errors="coerce").max()
        )
    else:
        df["or_pct_in_race"] = np.nan

    # --- Form string features ---
    form_col = "past_performance" if "past_performance" in df.columns else None
    if form_col:
        form_parsed = df[form_col].apply(_parse_form).apply(pd.Series)
        for col in form_parsed.columns:
            df[col] = form_parsed[col]
    else:
        for col in ["form_win_rate", "form_place_rate", "form_recent_score",
                    "form_run_count", "form_last1"]:
            df[col] = np.nan
        df["form_had_dnf"] = False

    # --- Weight relative to field mean ---
    w_num = pd.to_numeric(df.get("weight"), errors="coerce")
    df["weight_relative"] = w_num - df.groupby("race_id")["weight"].transform(
        lambda x: pd.to_numeric(x, errors="coerce").mean()
    )

    # --- Type casts ---
    for col in ["OR", "TS", "RPR", "Flag1", "Flag2", "Flag3", "Flag4", "Flag5",
                "draw", "last_run", "horse_age", "weight", "trainer_RFT", "price_money"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- NA imputation strategy:
    #   OR / TS / RPR — left as NaN; XGBoost/LightGBM learn optimal split for missing
    #   Flag scores    — impute with race mean (they're derived, race mean is most meaningful)
    #   draw, age, weight, price — race mean → global median fallback
    #   last_run, trainer_RFT   — global median
    #   racewkday/racemonth     — mode

    _race_mean_cols = ["Flag1", "Flag2", "Flag3", "Flag4", "Flag5",
                       "draw", "horse_age", "weight", "weight_relative", "price_money"]
    for col in _race_mean_cols:
        if col not in df.columns:
            continue
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        race_mean = df.groupby("race_id")[col].transform("mean")
        df[col] = df[col].fillna(race_mean).fillna(df[col].median())

    for col in ["last_run", "trainer_RFT"]:
        if col in df.columns:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(df[col].median())

    for col in ["racewkday", "racemonth"]:
        if col in df.columns:
            mode = df[col].mode()
            df[col] = df[col].fillna(mode.iloc[0] if not mode.empty else 0)

    for col in ["flag4_z_in_race", "or_pct_in_race", "form_recent_score"]:
        if col in df.columns:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if "flag4_rank_in_race" in df.columns:
        df["flag4_rank_in_race"] = df["flag4_rank_in_race"].fillna(
            df.get("field_size", pd.Series(dtype=float)) / 2
        )

    for col in BOOLEAN_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype(bool)

    for col in CATEGORICAL_FEATURES:
        df[col] = df.get(col, pd.Series(dtype=str)).fillna("Unknown").astype(str)

    return df


# ---------------------------------------------------------------------------
# sklearn preprocessor
# ---------------------------------------------------------------------------

def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline([("scaler", StandardScaler())])
    categorical_pipeline = Pipeline([
        ("encoder", OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            dtype=np.float64,
        )),
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
            ("bool", "passthrough", BOOLEAN_FEATURES),
        ],
        remainder="drop",
    )
    return preprocessor


def fit_preprocessor(df: pd.DataFrame, save_path: str = "models/preprocessor.pkl") -> ColumnTransformer:
    preprocessor = build_preprocessor()
    preprocessor.fit(df[ALL_FEATURES])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(preprocessor, save_path)
    feature_path = save_path.replace("preprocessor.pkl", "feature_columns.json")
    with open(feature_path, "w") as f:
        json.dump(ALL_FEATURES, f, indent=2)
    logger.info(f"Preprocessor saved to {save_path}")
    return preprocessor


def load_preprocessor(path: str = "models/preprocessor.pkl") -> Optional[ColumnTransformer]:
    if not os.path.exists(path):
        return None
    return joblib.load(path)


def transform(df: pd.DataFrame, preprocessor: ColumnTransformer) -> np.ndarray:
    return preprocessor.transform(df[ALL_FEATURES])


def get_feature_names() -> list:
    return ALL_FEATURES
