"""
Horse Race Prediction — Streamlit App
"""

import logging
import os
from datetime import date

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Horse Race Predictor",
    page_icon="🏇",
    layout="wide",
    initial_sidebar_state="expanded",
)

logging.basicConfig(level=logging.INFO)

PAGES = [
    "📋 Today's Picks",
    "🤖 ML Predictions",
    "🔍 Race Detail",
    "📅 Yesterday's Results",
    "⚙️ Train",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _append_to_csv(new_df: pd.DataFrame, path: str, dedup_keys: list) -> int:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        existing = pd.read_csv(path, low_memory=False)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=dedup_keys, keep="last")
    else:
        combined = new_df.drop_duplicates(subset=dedup_keys, keep="last")
    combined.to_csv(path, index=False)
    return len(combined)


def _init_state():
    defaults = {
        "racecard_df": None,
        "flag_picks": None,
        "ml_predictions": None,
        "models": None,
        "results_df": None,
        "scrape_error": None,
        "results_error": None,
        "train_metrics": None,
        "train_error": None,
        "today_timetable": None,
        "today_courses": [],
        "selected_courses": [],
        "page": PAGES[0],
        "prev_page": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


def _go(page: str):
    st.session_state["prev_page"] = st.session_state["page"]
    st.session_state["page"] = page
    st.rerun()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🏇 Horse Race Predictor")

    # Navigation
    nav = st.radio("Navigate", PAGES, index=PAGES.index(st.session_state["page"]), label_visibility="collapsed")
    if nav != st.session_state["page"]:
        _go(nav)

    # Home + Back buttons
    b1, b2 = st.columns(2)
    if b1.button("🏠 Home", use_container_width=True):
        _go(PAGES[0])
    if b2.button("◀ Back", use_container_width=True, disabled=st.session_state["prev_page"] is None):
        _go(st.session_state["prev_page"])

    st.markdown("---")

    # --- Step 1: fetch today's course list ---
    if st.button("📋 Fetch Today's Courses", use_container_width=True):
        with st.spinner("Fetching today's UK race schedule…"):
            try:
                from src.scraper import scrape_time_order
                timetable = scrape_time_order()
                if timetable.empty:
                    st.warning("No UK races found for today.")
                    st.session_state["today_courses"] = []
                    st.session_state["today_timetable"] = None
                else:
                    courses = sorted(timetable["racecourse"].dropna().unique().tolist())
                    st.session_state["today_courses"] = courses
                    st.session_state["selected_courses"] = courses
                    st.session_state["today_timetable"] = timetable
                    st.success(f"Found {len(timetable)} races at {len(courses)} course(s).")
            except Exception as e:
                st.error(f"Failed to fetch schedule: {e}")

    # --- Step 2: course picker ---
    if st.session_state["today_courses"]:
        st.session_state["selected_courses"] = st.multiselect(
            "Courses to scrape",
            options=st.session_state["today_courses"],
            default=st.session_state["selected_courses"],
            key="course_picker",
        )

    # --- Step 3: scrape selected courses ---
    scrape_ready = bool(
        st.session_state.get("today_timetable") is not None
        and st.session_state.get("selected_courses")
    )
    if st.button("🔄 Scrape Selected Races", use_container_width=True, disabled=not scrape_ready):
        timetable = st.session_state["today_timetable"]
        selected = st.session_state["selected_courses"]
        url_list = (
            timetable[timetable["racecourse"].isin(selected)]["meeting_url"].tolist()
            if timetable is not None and selected else None
        )
        with st.spinner(f"Scraping {len(url_list or [])} races…"):
            try:
                from src.scraper import scrape_all_races_today
                from src.flags import get_top_picks

                prog = st.empty()

                def _progress_cb(current, total, racecourse):
                    prog.info(f"[{current}/{total}] {racecourse}")

                df = scrape_all_races_today(progress_callback=_progress_cb, url_list=url_list)
                prog.empty()

                if df.empty or "race_id" not in df.columns:
                    st.session_state["scrape_error"] = "No races found."
                else:
                    st.session_state["racecard_df"] = df
                    st.session_state["flag_picks"] = get_top_picks(df, n=3)
                    st.session_state["scrape_error"] = None
                    total = _append_to_csv(df, "data/historical/raceid.csv", ["race_id", "horse_name"])
                    st.success(f"Scraped {df['race_id'].nunique()} races, {len(df)} horses. History: {total} rows.")

                    from src.predict import models_available, load_models
                    if not models_available():
                        st.info("No trained models yet — train first in the ⚙️ Train tab.")
                    else:
                        try:
                            from src.features import prepare_features
                            from src.predict import predict_win_probabilities
                            if st.session_state["models"] is None:
                                st.session_state["models"] = load_models()
                            prep_df = prepare_features(df, england_only=False)
                            if prep_df.empty:
                                st.warning("ML skipped — no horses after feature prep (ratings likely all missing).")
                            else:
                                preds = predict_win_probabilities(prep_df, st.session_state["models"])
                                st.session_state["ml_predictions"] = preds
                                st.success(f"ML predictions ready — {preds['race_id'].nunique()} races.")
                        except Exception as e:
                            st.warning(f"ML prediction failed: {e}")
            except Exception as e:
                st.session_state["scrape_error"] = str(e)
                st.error(f"Scrape failed: {e}")

    if st.button("📊 Scrape Yesterday's Results", use_container_width=True):
        with st.spinner("Scraping yesterday's results…"):
            try:
                from src.results_scraper import scrape_yesterday_results
                results = scrape_yesterday_results()
                if results.empty:
                    st.session_state["results_error"] = "No results found for yesterday."
                else:
                    st.session_state["results_df"] = results
                    st.session_state["results_error"] = None
                    total = _append_to_csv(results, "data/historical/results.csv", ["race_id", "horse_name"])
                    st.success(f"Loaded {len(results)} results. History: {total} rows.")
            except Exception as e:
                st.session_state["results_error"] = str(e)
                st.error(f"Results scrape failed: {e}")

    st.markdown("---")

    # Filters
    df_all = st.session_state["racecard_df"]
    racecourse_filter = []
    time_filter = []
    if df_all is not None and not df_all.empty:
        courses = sorted(df_all["racecourse"].dropna().unique().tolist())
        racecourse_filter = st.multiselect("Filter racecourse", courses, default=courses)
        times = sorted(df_all["race_time"].dropna().unique().tolist())
        time_filter = st.multiselect("Filter race time", times, default=times)
    else:
        st.caption("Scrape today's races to enable filters.")

    st.markdown("---")
    st.caption(f"Date: {date.today()}")


# ---------------------------------------------------------------------------
# Filter helper
# ---------------------------------------------------------------------------

def _apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if racecourse_filter:
        df = df[df["racecourse"].isin(racecourse_filter)]
    if time_filter:
        df = df[df["race_time"].isin(time_filter)]
    return df


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

page = st.session_state["page"]

# ===========================================================================
# Today's Picks
# ===========================================================================

if page == PAGES[0]:
    st.header("Today's Top Picks — Flag4 Rule-Based")

    flag_picks = st.session_state.get("flag_picks")
    if flag_picks is None or flag_picks.empty:
        st.info("No data yet. Use **📋 Fetch Today's Courses → 🔄 Scrape Selected Races** in the sidebar.")
    else:
        filtered = _apply_filters(flag_picks)
        if filtered.empty:
            st.warning("No races match the current filters.")
        else:
            display_cols = [c for c in [
                "race_time", "racecourse", "going", "horse_name", "horse_no",
                "flag_rank", "Flag4", "Flag1", "Flag5", "OR", "TS", "RPR",
                "draw", "last_run", "horse_age", "weight",
            ] if c in filtered.columns]

            top1 = filtered[filtered["flag_rank"] == 1][
                ["race_time", "racecourse", "going", "horse_name", "Flag4"]
            ].copy()
            top1.columns = ["Time", "Course", "Going", "Top Pick", "Flag4 Score"]
            st.subheader("Top Pick Per Race")
            st.dataframe(top1, use_container_width=True, hide_index=True)

            st.subheader("All Top 3 Picks")
            st.dataframe(filtered[display_cols], use_container_width=True, hide_index=True)

            if st.button("🔍 View Race Detail →"):
                _go(PAGES[2])

# ===========================================================================
# ML Predictions
# ===========================================================================

elif page == PAGES[1]:
    st.header("ML Predictions — XGBoost + LightGBM Ensemble")

    from src.predict import models_available, load_metrics

    if not models_available():
        st.warning("No trained models yet.")
        st.markdown(
            "**Steps to get ML predictions:**\n"
            "1. Scrape races + results daily to build `data/historical/`\n"
            "2. Go to **⚙️ Train** and click **🚀 Train Models**\n"
            "3. Scrape today's races again — predictions will appear here"
        )
        if st.button("⚙️ Go to Train tab"):
            _go(PAGES[4])
    else:
        ml_preds = st.session_state.get("ml_predictions")
        if ml_preds is None or ml_preds.empty:
            st.info("Models are ready. Scrape today's races (sidebar) to generate predictions.")
        else:
            filtered_ml = _apply_filters(ml_preds)
            if filtered_ml.empty:
                st.warning("No races match the current filters.")
            else:
                metrics = load_metrics()
                if metrics:
                    with st.expander("Model Performance (validation set)"):
                        cols = st.columns(3)
                        for i, (name, m) in enumerate(metrics.items()):
                            with cols[i % 3]:
                                st.metric(f"{name.upper()} Log Loss", f"{m['log_loss']:.4f}")
                                st.metric(f"{name.upper()} AUC", f"{m['roc_auc']:.4f}")
                                st.metric(f"{name.upper()} Rank Acc.", f"{m['rank_accuracy']:.1%}")

                top_ml_cols = [c for c in [
                    "race_time", "racecourse", "going", "horse_name", "horse_no",
                    "ml_rank", "win_probability", "implied_odds",
                    "prob_xgb", "prob_lgbm", "is_selection", "Flag4", "OR", "TS", "RPR",
                ] if c in filtered_ml.columns]

                if "ml_rank" not in filtered_ml.columns:
                    filtered_ml = filtered_ml.copy()
                    filtered_ml["ml_rank"] = (
                        filtered_ml.groupby("race_id")["win_probability"]
                        .rank(ascending=False, method="first", na_option="bottom")
                        .astype(int)
                    )

                top5 = filtered_ml[filtered_ml["ml_rank"] <= 5].copy()

                def _colour_prob(val):
                    if val >= 0.35:
                        return "background-color: #2d6a2d; color: white"
                    elif val >= 0.20:
                        return "background-color: #6a5a2d; color: white"
                    return ""

                st.subheader("Top 5 Per Race")
                available_cols = [c for c in top_ml_cols if c in top5.columns]
                styled = top5[available_cols].style.applymap(
                    _colour_prob,
                    subset=["win_probability"] if "win_probability" in available_cols else [],
                )
                st.dataframe(styled, use_container_width=True, hide_index=True)

                if st.button("🔍 View Race Detail →"):
                    _go(PAGES[2])

# ===========================================================================
# Race Detail
# ===========================================================================

elif page == PAGES[2]:
    st.header("Race Detail")

    df_all = st.session_state.get("racecard_df")
    ml_preds = st.session_state.get("ml_predictions")

    if df_all is None or df_all.empty:
        st.info("No data yet. Scrape today's races first.")
    else:
        filtered_all = _apply_filters(df_all)
        if filtered_all.empty:
            st.warning("No races match the current filters.")
        else:
            race_options = (
                filtered_all[["race_id", "race_time", "racecourse"]]
                .drop_duplicates("race_id")
                .sort_values("race_time")
            )
            race_labels = {
                row["race_id"]: f"{row['race_time']} — {row['racecourse']}"
                for _, row in race_options.iterrows()
            }

            selected_race_id = st.selectbox(
                "Select race",
                options=list(race_labels.keys()),
                format_func=lambda x: race_labels[x],
            )

            race_df = filtered_all[filtered_all["race_id"] == selected_race_id].copy()
            if "Flag4" not in race_df.columns:
                race_df["Flag4"] = None
            race_df["Flag4"] = pd.to_numeric(race_df["Flag4"], errors="coerce")
            race_df = race_df.sort_values("Flag4", ascending=False)

            col_left, col_right = st.columns(2)
            with col_left:
                st.subheader("Race Card")
                show_cols = [c for c in [
                    "horse_no", "horse_name", "Flag4", "Flag1", "Flag5",
                    "OR", "TS", "RPR", "draw", "last_run", "horse_age", "weight", "going",
                ] if c in race_df.columns]
                st.dataframe(race_df[show_cols], use_container_width=True, hide_index=True)

            with col_right:
                st.subheader("Flag4 Scores")
                chart_df = race_df[["horse_name", "Flag4"]].dropna()
                if not chart_df.empty:
                    fig = px.bar(
                        chart_df.sort_values("Flag4"),
                        x="Flag4", y="horse_name", orientation="h",
                        color="Flag4", color_continuous_scale="Greens",
                        title=f"Flag4 — {race_labels.get(selected_race_id, '')}",
                    )
                    fig.update_layout(showlegend=False, coloraxis_showscale=False)
                    st.plotly_chart(fig, use_container_width=True)

            if ml_preds is not None and not ml_preds.empty:
                race_ml = ml_preds[ml_preds["race_id"] == selected_race_id].copy()
                if not race_ml.empty:
                    st.subheader("ML Win Probabilities")
                    fig2 = px.bar(
                        race_ml.sort_values("win_probability"),
                        x="win_probability", y="horse_name", orientation="h",
                        color="win_probability", color_continuous_scale="Blues",
                        title="ML Ensemble Win Probability",
                    )
                    fig2.update_layout(showlegend=False, coloraxis_showscale=False)
                    st.plotly_chart(fig2, use_container_width=True)

# ===========================================================================
# Yesterday's Results
# ===========================================================================

elif page == PAGES[3]:
    st.header("Yesterday's Results — Prediction Accuracy")

    if st.session_state.get("results_error"):
        st.error(st.session_state["results_error"])

    results_df = st.session_state.get("results_df")
    if results_df is None or results_df.empty:
        st.info("Click **📊 Scrape Yesterday's Results** in the sidebar.")
    else:
        st.subheader("Race Results")
        sort_col = "race_time" if "race_time" in results_df.columns else None
        st.dataframe(
            results_df.sort_values([sort_col, "result_pos"]) if sort_col else results_df,
            use_container_width=True, hide_index=True,
        )

        saved_pred_path = "data/output/predictions_yesterday.csv"
        if os.path.exists(saved_pred_path):
            st.subheader("Prediction Accuracy")
            yesterday_preds = pd.read_csv(saved_pred_path)
            from src.predict import join_predictions_with_results, compute_hit_rates
            merged = join_predictions_with_results(yesterday_preds, results_df)
            hit_rates = compute_hit_rates(merged)

            cols = st.columns(4)
            for col, key, label in [
                (cols[0], "flag4_win_rate", "Flag4 Win Rate"),
                (cols[1], "flag4_top3_rate", "Flag4 Top-3"),
                (cols[2], "ml_win_rate", "ML Win Rate"),
                (cols[3], "ml_top3_rate", "ML Top-3"),
            ]:
                if hit_rates.get(key) is not None:
                    col.metric(label, f"{hit_rates[key]:.1%}")

            st.dataframe(
                merged[[c for c in merged.columns if not c.startswith("_")]],
                use_container_width=True, hide_index=True,
            )
        else:
            st.info("No saved predictions found. Today's predictions are saved automatically for tomorrow.")

# ===========================================================================
# Train
# ===========================================================================

elif page == PAGES[4]:
    st.header("Train Models")

    CARDS_PATH = "data/historical/raceid.csv"
    RESULTS_PATH = "data/historical/results.csv"
    cards_exist = os.path.exists(CARDS_PATH)
    results_exist = os.path.exists(RESULTS_PATH)

    col_a, col_b = st.columns(2)
    with col_a:
        if cards_exist:
            n = len(pd.read_csv(CARDS_PATH, usecols=[0]))
            st.success(f"raceid.csv — {n:,} rows")
        else:
            st.warning("raceid.csv not found in data/historical/")
    with col_b:
        if results_exist:
            n = len(pd.read_csv(RESULTS_PATH, usecols=[0]))
            st.success(f"results.csv — {n:,} rows")
        else:
            st.warning("results.csv not found in data/historical/")

    st.caption("Both files grow automatically each time you use the scrape buttons.")
    st.markdown("---")

    if not cards_exist or not results_exist:
        st.info("Scrape races and results first to build the training files.")
    else:
        if st.button("🚀 Train Models", use_container_width=True):
            with st.spinner("Training XGBoost + LightGBM… this may take a few minutes."):
                try:
                    from src.train import train
                    metrics = train()
                    st.session_state["train_metrics"] = metrics
                    st.session_state["train_error"] = None
                    from src.predict import load_models
                    st.session_state["models"] = load_models()
                    st.success("Training complete! Models saved to models/")
                except Exception as e:
                    st.session_state["train_error"] = str(e)
                    st.error(f"Training failed: {e}")

    if st.session_state.get("train_metrics"):
        st.subheader("Training Results")
        metrics = st.session_state["train_metrics"]
        cols = st.columns(3)
        for i, (name, m) in enumerate(metrics.items()):
            with cols[i % 3]:
                st.markdown(f"**{name.upper()}**")
                st.metric("Log Loss", f"{m['log_loss']:.4f}")
                st.metric("ROC-AUC", f"{m['roc_auc']:.4f}")
                st.metric("Brier Score", f"{m['brier_score']:.4f}")
                st.metric("Rank Accuracy", f"{m['rank_accuracy']:.1%}")

    if st.session_state.get("train_error"):
        st.error(st.session_state["train_error"])

# ---------------------------------------------------------------------------
# Auto-save predictions
# ---------------------------------------------------------------------------

if st.session_state.get("ml_predictions") is not None:
    preds = st.session_state["ml_predictions"]
    if not preds.empty:
        os.makedirs("data/output", exist_ok=True)
        preds.to_csv("data/output/predictions_yesterday.csv", index=False)
