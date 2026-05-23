"""
Horse Race Prediction — Streamlit App

Pages:
1. Today's Picks (Flag)  — rule-based Flag4 top picks
2. ML Predictions        — XGBoost + LightGBM ensemble probabilities
3. Race Detail           — per-race breakdown with bar chart
4. Yesterday's Results   — prediction accuracy vs actuals
5. Train / Evaluate      — train models from historical data
"""

import logging
import os
from datetime import date, timedelta

import pandas as pd
import plotly.express as px
import streamlit as st

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Horse Race Predictor",
    page_icon="🏇",
    layout="wide",
    initial_sidebar_state="expanded",
)

logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------

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
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🏇 Horse Race Predictor")
    st.markdown("---")

    # --- Scrape today ---
    if st.button("🔄 Scrape Today's Races", use_container_width=True):
        with st.spinner("Scraping today's racecards from RacingPost…"):
            try:
                from src.scraper import scrape_all_races_today
                from src.flags import build_today_predictions, get_top_picks

                prog_placeholder = st.empty()

                def progress_cb(current, total, racecourse):
                    prog_placeholder.info(f"[{current}/{total}] {racecourse}")

                df = scrape_all_races_today(progress_callback=progress_cb)
                prog_placeholder.empty()

                if df.empty or "race_id" not in df.columns:
                    st.session_state["scrape_error"] = (
                        "No races found for today. RacingPost may have updated their page "
                        "structure, or there are no races scheduled."
                    )
                else:
                    st.session_state["racecard_df"] = df
                    st.session_state["flag_picks"] = get_top_picks(df, n=3)
                    st.session_state["scrape_error"] = None
                    st.success(f"Scraped {df['race_id'].nunique()} races, {len(df)} horses.")

                    # Try ML predictions if models available
                    from src.predict import models_available, load_models
                    if models_available():
                        try:
                            from src.features import prepare_features
                            from src.predict import predict_win_probabilities

                            if st.session_state["models"] is None:
                                st.session_state["models"] = load_models()

                            prep_df = prepare_features(df)
                            if not prep_df.empty:
                                preds = predict_win_probabilities(prep_df, st.session_state["models"])
                                st.session_state["ml_predictions"] = preds
                        except Exception as e:
                            st.warning(f"ML prediction failed: {e}")
            except Exception as e:
                st.session_state["scrape_error"] = str(e)
                st.error(f"Scrape failed: {e}")

    # --- Scrape yesterday's results ---
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
                    st.success(f"Loaded {len(results)} results.")
            except Exception as e:
                st.session_state["results_error"] = str(e)
                st.error(f"Results scrape failed: {e}")

    st.markdown("---")

    # --- Filters (only shown when data is available) ---
    df_all = st.session_state["racecard_df"]
    racecourse_filter = []
    time_filter = []

    if df_all is not None and not df_all.empty:
        courses = sorted(df_all["racecourse"].dropna().unique().tolist())
        racecourse_filter = st.multiselect("Filter racecourse", courses, default=courses)

        times = sorted(df_all["race_time"].dropna().unique().tolist())
        time_filter = st.multiselect("Filter race time", times, default=times)
    else:
        st.info("Click 'Scrape Today's Races' to load data.")

    st.markdown("---")
    st.caption(f"Date: {date.today()}")

# ---------------------------------------------------------------------------
# Shared filtering helper
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
# Page tabs
# ---------------------------------------------------------------------------

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Today's Picks (Flag)",
    "🤖 ML Predictions",
    "🔍 Race Detail",
    "📅 Yesterday's Results",
    "⚙️ Train / Evaluate",
])

# ===========================================================================
# Tab 1 — Today's Picks (Flag-based)
# ===========================================================================

with tab1:
    st.header("Today's Top Picks — Flag4 Rule-Based")

    flag_picks = st.session_state.get("flag_picks")
    if flag_picks is None or flag_picks.empty:
        st.info("No data yet. Click 'Scrape Today's Races' in the sidebar.")
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

            # Summary: top pick per race
            top1 = filtered[filtered["flag_rank"] == 1][
                ["race_time", "racecourse", "going", "horse_name", "Flag4"]
            ].copy()
            top1.columns = ["Time", "Course", "Going", "Top Pick (Flag4)", "Flag4 Score"]

            st.subheader("Top Pick Per Race")
            st.dataframe(top1, use_container_width=True, hide_index=True)

            st.subheader("All Top 3 Picks")
            st.dataframe(filtered[display_cols], use_container_width=True, hide_index=True)

# ===========================================================================
# Tab 2 — ML Predictions
# ===========================================================================

with tab2:
    st.header("ML Predictions — XGBoost + LightGBM Ensemble")

    from src.predict import models_available, load_metrics

    if not models_available():
        st.warning(
            "No trained models found. Go to the **Train / Evaluate** tab to train models first, "
            "or run: `python src/train.py`"
        )
    else:
        ml_preds = st.session_state.get("ml_predictions")

        if ml_preds is None or ml_preds.empty:
            st.info("ML predictions will appear here after scraping today's races.")
        else:
            filtered_ml = _apply_filters(ml_preds)

            if filtered_ml.empty:
                st.warning("No races match the current filters.")
            else:
                # Show model metrics in expander
                metrics = load_metrics()
                if metrics:
                    with st.expander("Model Performance (validation set)"):
                        cols = st.columns(3)
                        for i, (model_name, m) in enumerate(metrics.items()):
                            with cols[i % 3]:
                                st.metric(f"{model_name.upper()} Log Loss", f"{m['log_loss']:.4f}")
                                st.metric(f"{model_name.upper()} AUC", f"{m['roc_auc']:.4f}")
                                st.metric(f"{model_name.upper()} Rank Acc.", f"{m['rank_accuracy']:.1%}")

                # Top picks by ML
                top_ml_cols = [c for c in [
                    "race_time", "racecourse", "going", "horse_name", "horse_no",
                    "ml_rank", "win_probability", "implied_odds",
                    "prob_xgb", "prob_lgbm", "is_selection",
                    "Flag4", "OR", "TS", "RPR",
                ] if c in filtered_ml.columns]

                # Add ml_rank if not present
                if "ml_rank" not in filtered_ml.columns:
                    filtered_ml = filtered_ml.copy()
                    filtered_ml["ml_rank"] = filtered_ml.groupby("race_id")["win_probability"].rank(
                        ascending=False, method="first"
                    ).astype(int)

                top5 = filtered_ml[filtered_ml["ml_rank"] <= 5].copy()

                # Colour win_probability column
                def _colour_prob(val):
                    if val >= 0.35:
                        return "background-color: #2d6a2d; color: white"
                    elif val >= 0.20:
                        return "background-color: #6a5a2d; color: white"
                    return ""

                st.subheader("Top 5 Per Race (ML Ensemble)")
                available_top5_cols = [c for c in top_ml_cols if c in top5.columns]
                styled = top5[available_top5_cols].style.applymap(
                    _colour_prob, subset=["win_probability"] if "win_probability" in available_top5_cols else []
                )
                st.dataframe(styled, use_container_width=True, hide_index=True)

# ===========================================================================
# Tab 3 — Race Detail
# ===========================================================================

with tab3:
    st.header("Race Detail")

    df_all = st.session_state.get("racecard_df")
    ml_preds = st.session_state.get("ml_predictions")

    if df_all is None or df_all.empty:
        st.info("No data yet. Click 'Scrape Today's Races' in the sidebar.")
    else:
        filtered_all = _apply_filters(df_all)

        if filtered_all.empty:
            st.warning("No races match the current filters.")
        else:
            # Race selector
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

            col_left, col_right = st.columns([1, 1])

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
                        x="Flag4",
                        y="horse_name",
                        orientation="h",
                        color="Flag4",
                        color_continuous_scale="Greens",
                        title=f"Flag4 Score — {race_labels.get(selected_race_id, '')}",
                    )
                    fig.update_layout(showlegend=False, coloraxis_showscale=False)
                    st.plotly_chart(fig, use_container_width=True)

            # ML overlay for this race (if available)
            if ml_preds is not None and not ml_preds.empty:
                race_ml = ml_preds[ml_preds["race_id"] == selected_race_id].copy()
                if not race_ml.empty:
                    st.subheader("ML Win Probabilities")
                    fig2 = px.bar(
                        race_ml.sort_values("win_probability"),
                        x="win_probability",
                        y="horse_name",
                        orientation="h",
                        color="win_probability",
                        color_continuous_scale="Blues",
                        title="ML Ensemble Win Probability",
                    )
                    fig2.update_layout(showlegend=False, coloraxis_showscale=False)
                    st.plotly_chart(fig2, use_container_width=True)

# ===========================================================================
# Tab 4 — Yesterday's Results
# ===========================================================================

with tab4:
    st.header("Yesterday's Results — Prediction Accuracy")

    results_df = st.session_state.get("results_df")
    flag_picks = st.session_state.get("flag_picks")

    if st.session_state.get("results_error"):
        st.error(st.session_state["results_error"])

    if results_df is None or results_df.empty:
        st.info("Click 'Scrape Yesterday's Results' in the sidebar.")
    else:
        st.subheader("Yesterday's Race Results")
        st.dataframe(
            results_df.sort_values(["race_time", "result_pos"]) if "race_time" in results_df.columns
            else results_df,
            use_container_width=True,
            hide_index=True,
        )

        # If we have saved predictions for yesterday, compute accuracy
        saved_pred_path = "data/output/predictions_yesterday.csv"
        if os.path.exists(saved_pred_path):
            st.subheader("Prediction Accuracy")
            yesterday_preds = pd.read_csv(saved_pred_path)

            from src.predict import join_predictions_with_results, compute_hit_rates

            merged = join_predictions_with_results(yesterday_preds, results_df)
            hit_rates = compute_hit_rates(merged)

            cols = st.columns(4)
            if "flag4_win_rate" in hit_rates and hit_rates["flag4_win_rate"] is not None:
                cols[0].metric("Flag4 Win Rate", f"{hit_rates['flag4_win_rate']:.1%}")
            if "flag4_top3_rate" in hit_rates and hit_rates["flag4_top3_rate"] is not None:
                cols[1].metric("Flag4 Top-3 Rate", f"{hit_rates['flag4_top3_rate']:.1%}")
            if "ml_win_rate" in hit_rates and hit_rates["ml_win_rate"] is not None:
                cols[2].metric("ML Win Rate", f"{hit_rates['ml_win_rate']:.1%}")
            if "ml_top3_rate" in hit_rates and hit_rates["ml_top3_rate"] is not None:
                cols[3].metric("ML Top-3 Rate", f"{hit_rates['ml_top3_rate']:.1%}")

            st.dataframe(
                merged[[c for c in merged.columns if not c.startswith("_")]],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info(
                "No saved predictions for yesterday found. "
                "Today's predictions will be saved automatically for tomorrow's comparison."
            )

# ===========================================================================
# Tab 5 — Train / Evaluate
# ===========================================================================

with tab5:
    st.header("Train Models")
    st.markdown(
        "Upload historical racecard and results CSV files to train the XGBoost + LightGBM models."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Upload historical racecards")
        uploaded_cards = st.file_uploader(
            "raceid.csv (historical racecards)",
            type="csv",
            key="upload_cards",
        )
        if uploaded_cards:
            os.makedirs("data/historical", exist_ok=True)
            with open("data/historical/raceid.csv", "wb") as f:
                f.write(uploaded_cards.getbuffer())
            st.success("raceid.csv saved.")

    with col2:
        st.subheader("Upload historical results")
        uploaded_results = st.file_uploader(
            "results.csv (historical results)",
            type="csv",
            key="upload_results",
        )
        if uploaded_results:
            os.makedirs("data/historical", exist_ok=True)
            with open("data/historical/results.csv", "wb") as f:
                f.write(uploaded_results.getbuffer())
            st.success("results.csv saved.")

    st.markdown("---")

    cards_exist = os.path.exists("data/historical/raceid.csv")
    results_exist = os.path.exists("data/historical/results.csv")

    if not cards_exist or not results_exist:
        st.warning(
            "Both `data/historical/raceid.csv` and `data/historical/results.csv` are required. "
            "Upload them above or place them in the `data/historical/` folder."
        )
    else:
        st.success("Historical data found. Ready to train.")

        if st.button("🚀 Train Models", use_container_width=True):
            with st.spinner("Training XGBoost + LightGBM… this may take a few minutes."):
                try:
                    from src.train import train
                    metrics = train()
                    st.session_state["train_metrics"] = metrics
                    st.session_state["train_error"] = None
                    # Reload models
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
        for i, (model_name, m) in enumerate(metrics.items()):
            with cols[i % 3]:
                st.markdown(f"**{model_name.upper()}**")
                st.metric("Log Loss", f"{m['log_loss']:.4f}")
                st.metric("ROC-AUC", f"{m['roc_auc']:.4f}")
                st.metric("Brier Score", f"{m['brier_score']:.4f}")
                st.metric("Rank Accuracy", f"{m['rank_accuracy']:.1%}")

    if st.session_state.get("train_error"):
        st.error(st.session_state["train_error"])

# ---------------------------------------------------------------------------
# Auto-save today's predictions for tomorrow's accuracy check
# ---------------------------------------------------------------------------

if st.session_state.get("ml_predictions") is not None:
    preds = st.session_state["ml_predictions"]
    if not preds.empty:
        os.makedirs("data/output", exist_ok=True)
        preds.to_csv("data/output/predictions_yesterday.csv", index=False)
