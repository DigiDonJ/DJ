"""
Flag1–Flag5 scoring system + race factor computation.

Preserves the user's original R flag logic with a fixed sort order for factor
calculations (R's nth() was not sorting first — corrected here).
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Flag computation (already done during scraping, but can recompute here)
# ---------------------------------------------------------------------------

def compute_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Flag1–Flag5 columns from OR, TS, RPR, weight.

    - Flag1 = TS + RPR − OR          (form vs official rating)
    - Flag2 = TS − weight            (topspeed adjusted for weight)
    - Flag3 = RPR − weight           (RP rating adjusted for weight)
    - Flag4 = (Flag1×2)+(Flag2×1)+(Flag3×1.75)  ← composite (primary sort key)
    - Flag5 = (Flag1+Flag2+Flag3)/3  ← simple average
    """
    df = df.copy()

    or_ = pd.to_numeric(df.get("OR"), errors="coerce")
    ts = pd.to_numeric(df.get("TS"), errors="coerce")
    rpr = pd.to_numeric(df.get("RPR"), errors="coerce")
    weight = pd.to_numeric(df.get("weight"), errors="coerce")

    df["Flag1"] = ts + rpr - or_
    df["Flag2"] = ts - weight
    df["Flag3"] = rpr - weight
    df["Flag4"] = (df["Flag1"] * 2) + (df["Flag2"] * 1) + (df["Flag3"] * 1.75)
    df["Flag5"] = (df["Flag1"] + df["Flag2"] + df["Flag3"]) / 3

    return df


# ---------------------------------------------------------------------------
# Race-level factor scores
# ---------------------------------------------------------------------------

def _nth_sorted(series: pd.Series, n: int) -> float:
    """Return the nth largest value in a series (1-indexed). Fixed from R's nth()."""
    vals = series.dropna().sort_values(ascending=False)
    if len(vals) < n:
        return np.nan
    return float(vals.iloc[n - 1])


def compute_race_factors(df: pd.DataFrame) -> pd.Series:
    """
    Compute summary factor scores for a single race's DataFrame.

    Returns a dict/Series with:
        Top_Factor, Fst/Snd/Trd_Flg1_Fac, Fst/Snd/Trd_Flg4_Fac, Fst/Snd/Trd_Flg5_Fac
    """
    n = len(df)
    if n == 0:
        return pd.Series(dtype=float)

    def factor(flag_col: str) -> tuple:
        vals = pd.to_numeric(df[flag_col], errors="coerce").dropna()
        if vals.empty:
            return np.nan, np.nan, np.nan
        mean = vals.mean()
        top1 = _nth_sorted(vals, 1)
        top2 = _nth_sorted(vals, 2)
        top3 = _nth_sorted(vals, 3)
        fst = round((top1 - mean) / n, 2)
        snd = round((top1 - top2) / n, 2) if not np.isnan(top2) else np.nan
        trd = round((top2 - top3) / n, 2) if not np.isnan(top3) else np.nan
        return fst, snd, trd

    flag4_fst, flag4_snd, flag4_trd = factor("Flag4")
    flag1_fst, flag1_snd, flag1_trd = factor("Flag1")
    flag5_fst, flag5_snd, flag5_trd = factor("Flag5")

    return pd.Series({
        "Top_Factor": flag4_fst,
        "Fst_Flg1_Fac": flag1_fst,
        "Snd_Flg1_Fac": flag1_snd,
        "Trd_Flg1_Fac": flag1_trd,
        "Fst_Flg4_Fac": flag4_fst,
        "Snd_Flg4_Fac": flag4_snd,
        "Trd_Flg4_Fac": flag4_trd,
        "Fst_Flg5_Fac": flag5_fst,
        "Snd_Flg5_Fac": flag5_snd,
        "Trd_Flg5_Fac": flag5_trd,
    })


# ---------------------------------------------------------------------------
# Top picks per race
# ---------------------------------------------------------------------------

def get_top_picks(df: pd.DataFrame, n: int = 3, sort_by: str = "Flag4") -> pd.DataFrame:
    """
    Return the top n horses per race, ranked by sort_by flag.

    Args:
        df: full racecard DataFrame (must have race_id + flag columns)
        n: number of top horses to return per race
        sort_by: flag column to rank by (default: Flag4)

    Returns:
        DataFrame with top n per race, with added column 'flag_rank' (1=top pick)
    """
    df = df.copy()
    df[sort_by] = pd.to_numeric(df[sort_by], errors="coerce")

    # Rank within each race (1 = highest flag score)
    df["flag_rank"] = df.groupby("race_id")[sort_by].rank(ascending=False, method="first").astype(int)

    top = df[df["flag_rank"] <= n].copy()
    top = top.sort_values(["race_time", "race_id", "flag_rank"]).reset_index(drop=True)
    return top


def build_today_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the TodayallToppred-equivalent summary table.

    For each race: top 3 horses by Flag1, Flag4, Flag5 + race factor scores.
    Returns one row per race.
    """
    if df.empty:
        return pd.DataFrame()

    rows = []
    for race_id, race_df in df.groupby("race_id"):
        race_df = race_df.copy()
        factors = compute_race_factors(race_df)

        def top3(col):
            sub = (
                race_df[["horse_no", "horse_name", col]]
                .dropna(subset=[col])
                .sort_values(col, ascending=False)
                .head(3)
            )
            names = sub["horse_name"].tolist()
            nos = sub["horse_no"].tolist()
            return names, nos

        fl1_names, fl1_nos = top3("Flag1")
        fl4_names, fl4_nos = top3("Flag4")
        fl5_names, fl5_nos = top3("Flag5")

        def _get(lst, i):
            return lst[i] if i < len(lst) else None

        row = {
            **factors.to_dict(),
            "race_id": race_id,
            "racecourse": race_df["racecourse"].iloc[0],
            "race_date": race_df["race_date"].iloc[0],
            "race_time": race_df["race_time"].iloc[0],
            "going": race_df["going"].iloc[0],
            "num_runners": race_df["num_runners"].iloc[0],
            "FL1_no_1": _get(fl1_nos, 0), "FL1_name_1": _get(fl1_names, 0),
            "FL1_no_2": _get(fl1_nos, 1), "FL1_name_2": _get(fl1_names, 1),
            "FL1_no_3": _get(fl1_nos, 2), "FL1_name_3": _get(fl1_names, 2),
            "FL4_no_1": _get(fl4_nos, 0), "FL4_name_1": _get(fl4_names, 0),
            "FL4_no_2": _get(fl4_nos, 1), "FL4_name_2": _get(fl4_names, 1),
            "FL4_no_3": _get(fl4_nos, 2), "FL4_name_3": _get(fl4_names, 2),
            "FL5_no_1": _get(fl5_nos, 0), "FL5_name_1": _get(fl5_names, 0),
            "FL5_no_2": _get(fl5_nos, 1), "FL5_name_2": _get(fl5_names, 1),
            "FL5_no_3": _get(fl5_nos, 2), "FL5_name_3": _get(fl5_names, 2),
        }
        rows.append(row)

    return pd.DataFrame(rows)
