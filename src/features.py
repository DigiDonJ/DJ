"""
Feature engineering pipeline for the horse race prediction ML models.

Ports the user's R caret preprocessing to a sklearn ColumnTransformer,
adding race-relative features (flag rank, z-score, OR percentile).
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
    "OR", "TS", "RPR",
    "Flag1", "Flag2", "Flag3", "Flag4", "Flag5",
    "draw", "last_run", "horse_age", "weight",
    "price_money", "racewkday", "racemonth",
    "trainer_RFT",
    "field_size",
    "flag4_rank_in_race",
    "flag4_z_in_race",
    "or_pct_in_race",
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
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BOOLEAN_FEATURES

TARGET_COLUMN = "won"


# ---------------------------------------------------------------------------
# Data preparation (from R's TDfiltered_England pipeline)
# ---------------------------------------------------------------------------

def prepare_features(
    df: pd.DataFrame,
    coursecountry_path: str = "data/Coursecountry.csv",
    england_only: bool = True,
) -> pd.DataFrame:
    """
    Prepare features for ML modelling.

    Steps:
    1. Join with Coursecountry lookup
    2. Filter to England (if england_only=True)
    3. Add date-derived features
    4. Add missing-value indicators
    5. Add race-relative features
    6. Cast types
    7. Select feature columns (+ target if present)
    """
    df = df.copy()

    # --- Normalize column names from scraper ---
    col_map = {
        "horse_name": "horse_name",
        "price_money": "price_money",
        "last_run": "last_run",
        "horse_age": "horse_age",
        "trainer_RFT": "trainer_RFT",
        "jockey_allowance": "jockey_allowance",
        "num_runners": "num_runners",
        "race_date": "race_date",
        "race_time": "race_time",
        "race_id": "race_id",
        "racecourse": "racecourse",
    }

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

    # --- Compute Flag1-5 if not already present (historical R CSVs have raw ratings only) ---
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

    # --- Drop rows where Flag4 cannot be computed at all ---
    df = df.dropna(subset=["Flag4"])

    # --- Date features ---
    dates = pd.to_datetime(df["race_date"], format="%Y-%m-%d", errors="coerce")
    df["racemonth"] = dates.dt.month
    df["racewkday"] = dates.dt.dayofweek  # 0=Monday, 6=Sunday

    # --- Missing value indicators ---
    df["missing_trainerRFT"] = df["trainer_RFT"].isna()
    df["missing_draw"] = df["draw"].fillna(0) == 0
    df["missing_lastrun"] = df["last_run"].isna()

    # --- Field size ---
    df["field_size"] = df.groupby("race_id")["race_id"].transform("count")

    # --- Race-relative features ---
    df["Flag4_num"] = pd.to_numeric(df["Flag4"], errors="coerce")
    df["flag4_rank_in_race"] = df.groupby("race_id")["Flag4_num"].rank(
        ascending=False, method="first"
    )
    df["flag4_z_in_race"] = df.groupby("race_id")["Flag4_num"].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-8)
    )
    or_num = pd.to_numeric(df["OR"], errors="coerce")
    df["or_pct_in_race"] = or_num / df.groupby("race_id")["OR"].transform(
        lambda x: pd.to_numeric(x, errors="coerce").max()
    )

    # --- Type casts ---
    for col in ["OR", "TS", "RPR", "Flag1", "Flag2", "Flag3", "draw",
                "last_run", "horse_age", "weight", "trainer_RFT"]:
        df[col] = pd.to_numeric(df.get(col), errors="coerce")

    df["Flag4"] = pd.to_numeric(df["Flag4"], errors="coerce")
    df["Flag5"] = pd.to_numeric(df["Flag5"], errors="coerce")
    df["price_money"] = pd.to_numeric(df.get("price_money"), errors="coerce")

    # --- NA imputation: race-level mean → global median fallback ---
    # Columns where a within-race average is the most meaningful substitute.
    _impute_with_race_mean = [
        "OR", "TS", "RPR", "Flag1", "Flag2", "Flag3", "Flag4", "Flag5",
        "draw", "horse_age", "weight", "price_money",
    ]
    for col in _impute_with_race_mean:
        if col not in df.columns:
            continue
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        race_mean = df.groupby("race_id")[col].transform("mean")
        global_median = df[col].median()
        df[col] = df[col].fillna(race_mean).fillna(global_median)

    # last_run and trainer_RFT: use global median (no within-race meaning)
    for col in ["last_run", "trainer_RFT"]:
        if col in df.columns:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            df[col] = df[col].fillna(df[col].median())

    # racewkday / racemonth: fill with mode (all today's races share the same date,
    # but if race_date failed to parse we'd get NaN)
    for col in ["racewkday", "racemonth"]:
        if col in df.columns:
            mode = df[col].mode()
            df[col] = df[col].fillna(mode.iloc[0] if not mode.empty else 0)

    # Race-relative derived features — clip inf then fill residual NAs
    for col in ["flag4_z_in_race", "or_pct_in_race"]:
        if col in df.columns:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if "flag4_rank_in_race" in df.columns:
        df["flag4_rank_in_race"] = df["flag4_rank_in_race"].fillna(
            df.get("field_size", pd.Series(dtype=float)) / 2
        )

    for col in BOOLEAN_FEATURES:
        df[col] = df[col].astype(bool)

    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].fillna("Unknown").astype(str)

    # --- Drop rows still missing both Flag4 and OR after imputation ---
    df = df.dropna(subset=["Flag4", "OR"])

    return df


# ---------------------------------------------------------------------------
# sklearn preprocessor: fit on training data, transform train + inference
# ---------------------------------------------------------------------------

def build_preprocessor() -> ColumnTransformer:
    """
    Build a sklearn ColumnTransformer with:
    - StandardScaler for numerics
    - OrdinalEncoder for categoricals (handles unseen values at inference)
    - Passthrough for booleans
    """
    numeric_pipeline = Pipeline([
        ("scaler", StandardScaler()),
    ])

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
    """Fit preprocessor on training data and save."""
    preprocessor = build_preprocessor()
    preprocessor.fit(df[ALL_FEATURES])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(preprocessor, save_path)

    # Save feature column list
    feature_path = save_path.replace("preprocessor.pkl", "feature_columns.json")
    with open(feature_path, "w") as f:
        json.dump(ALL_FEATURES, f, indent=2)

    logger.info(f"Preprocessor saved to {save_path}")
    return preprocessor


def load_preprocessor(path: str = "models/preprocessor.pkl") -> Optional[ColumnTransformer]:
    """Load a saved preprocessor."""
    if not os.path.exists(path):
        return None
    return joblib.load(path)


def transform(df: pd.DataFrame, preprocessor: ColumnTransformer) -> np.ndarray:
    """Apply preprocessor to a DataFrame, returning a numpy array."""
    return preprocessor.transform(df[ALL_FEATURES])


def get_feature_names() -> list:
    return ALL_FEATURES
