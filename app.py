"""
========================================================================
Streamlit App — Sentiment Analysis of Product Reviews
========================================================================
Run with: streamlit run streamlit_app.py

Requires:
  - saved_models/best_model.pkl
  - saved_models/tfidf_vectorizer.pkl
  - saved_models/feature_selector.pkl
  - saved_models/label_encoder.pkl
  - saved_models/metadata.json

Run the main pipeline first:
  python sentiment_analysis.py Amazon_Reviews.csv
========================================================================
"""

import os
import json
import streamlit as st
import pandas as pd

# Import inference helpers from main module
from sentiment_analysis import load_inference_pipeline, predict_sentiment

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Sentiment Analyser",
    page_icon="🔍",
    layout="wide"
)

PALETTE = {"Negative": "#E74C3C", "Neutral": "#F39C12", "Positive": "#27AE60"}

# ─────────────────────────────────────────────
# LOAD ARTIFACTS (cached)
# ─────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    return load_inference_pipeline()


@st.cache_data
def load_metadata():
    path = os.path.join("saved_models", "metadata.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


# ─────────────────────────────────────────────
# SIDEBAR — Model Info
# ─────────────────────────────────────────────
meta = load_metadata()

st.sidebar.title("🤖 Model Information")
if meta:
    st.sidebar.markdown(f"**Best Model**: `{meta.get('best_model', 'N/A')}`")
    st.sidebar.metric("Macro F1",  f"{meta.get('macro_f1', 0):.4f}")
    st.sidebar.metric("Accuracy",  f"{meta.get('accuracy', 0):.4f}")
    st.sidebar.metric("Precision", f"{meta.get('precision', 0):.4f}")
    st.sidebar.metric("Recall",    f"{meta.get('recall', 0):.4f}")
else:
    st.sidebar.warning("No metadata found.\nRun the pipeline first.")

# ─────────────────────────────────────────────
# MAIN PAGE
# ─────────────────────────────────────────────
st.title("🔍 Product Review Sentiment Analyser")
st.markdown(
    "Classify product reviews as **Positive**, **Neutral**, or **Negative** "
    "using TF-IDF + Machine Learning."
)

# ─── Tab Layout ────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Single Review", "Batch Analysis", "Saved Plots"])

# ── Tab 1: Single Review ────────────────────────────────────
with tab1:
    st.subheader("Analyse a Single Review")
    review_text = st.text_area(
        "Paste your product review below:",
        height=150,
        placeholder="e.g. 'This product is absolutely fantastic! Great quality and fast delivery.'"
    )

    if st.button("🔎 Analyse Sentiment", type="primary"):
        if not review_text.strip():
            st.warning("⚠️ Please enter a review text before analysing.")
        else:
            try:
                artifacts = load_artifacts()
                with st.spinner("Analysing…"):
                    result = predict_sentiment(review_text, artifacts)

                sentiment = result["sentiment"]
                color     = PALETTE.get(sentiment, "#888")

                emoji_map = {"Positive": "😊", "Neutral": "😐", "Negative": "😞"}
                emoji = emoji_map.get(sentiment, "🔍")

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(
                    f"""
                    <div style="background:{color}22; border: 2px solid {color};
                                padding:28px 20px; border-radius:12px;
                                text-align:center; margin-bottom:16px">
                        <div style="font-size:2.8rem; margin-bottom:6px">{emoji}</div>
                        <div style="color:{color}; font-size:1.8rem;
                                    font-weight:700; letter-spacing:1px">
                            {sentiment}
                        </div>
                        <div style="color:{color}99; font-size:0.9rem; margin-top:4px">
                            Predicted Sentiment
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                if result["probabilities"]:
                    st.markdown("#### 📊 Confidence Scores")
                    prob_df = pd.DataFrame(
                        result["probabilities"].items(),
                        columns=["Sentiment", "Probability"]
                    ).sort_values("Probability", ascending=False)

                    # Show colour-coded metrics
                    cols = st.columns(len(prob_df))
                    for col, (_, row) in zip(cols, prob_df.iterrows()):
                        em = emoji_map.get(row["Sentiment"], "")
                        col.metric(
                            label=f"{em} {row['Sentiment']}",
                            value=f"{row['Probability']*100:.1f}%"
                        )

                    st.markdown("<br>", unsafe_allow_html=True)
                    st.bar_chart(prob_df.set_index("Sentiment"))

            except FileNotFoundError:
                st.error("⚠️ Model artifacts not found. "
                         "Please run `python sentiment_analysis.py <your_csv>` first.")

# ── Tab 2: Batch Analysis ───────────────────────────────────
with tab2:
    st.subheader("Batch CSV Analysis")
    st.info("Upload a CSV with a column containing review texts.")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        df_upload = pd.read_csv(uploaded, on_bad_lines="skip", engine="python")
        st.write("**Preview:**", df_upload.head())

        text_cols = df_upload.select_dtypes(include="object").columns.tolist()
        col_sel   = st.selectbox("Select text column", text_cols)

        if st.button("Run Batch Analysis", type="primary"):
            try:
                artifacts = load_artifacts()
                predictions = []
                prog = st.progress(0)
                total = len(df_upload)

                for i, row in df_upload.iterrows():
                    raw = row[col_sel]
                    text = str(raw).strip() if pd.notna(raw) else ""
                    if not text or text.lower() in ("nan", "none", ""):
                        predictions.append("Neutral")   # safe fallback for blank rows
                    else:
                        res = predict_sentiment(text, artifacts)
                        predictions.append(res["sentiment"])
                    prog.progress((i + 1) / total)

                df_upload["Predicted Sentiment"] = predictions
                st.success(f"Analysed {total:,} reviews!")
                st.write(df_upload[[col_sel, "Predicted Sentiment"]].head(20))

                # Distribution
                dist = df_upload["Predicted Sentiment"].value_counts()
                st.bar_chart(dist)

                # Download
                csv_out = df_upload.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download Results CSV",
                    csv_out, "predictions.csv", "text/csv"
                )

            except FileNotFoundError:
                st.error("⚠️ Model artifacts not found.")

# ── Tab 3: Saved Plots ──────────────────────────────────────
with tab3:
    st.subheader("Training Visualisations")
    plots_dir = "plots"

    if os.path.isdir(plots_dir):
        plot_files = sorted(
            f for f in os.listdir(plots_dir) if f.endswith(".png")
        )
        if plot_files:
            for pf in plot_files:
                caption = pf.replace("_", " ").replace(".png", "").title()
                st.image(os.path.join(plots_dir, pf), caption=caption, use_column_width=True)
                st.markdown("---")
        else:
            st.info("No plots found. Run the pipeline to generate them.")
    else:
        st.info("Plots directory not found. Run the pipeline first.")
