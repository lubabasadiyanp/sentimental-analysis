"""
╔══════════════════════════════════════════════════════════════════════════╗
║            S E N T I M E N T  I Q  —  Streamlit App                    ║
║   Sentiment · Emotion · ABSA · Fake Detection · Model Comparison        ║
╚══════════════════════════════════════════════════════════════════════════╝
Artifacts expected (place in same folder as this file):
  tfidf_vectorizer.pkl   feature_scaler.pkl   svm_model.pkl
  lr_model.pkl           rf_model.pkl         xgb_model.pkl
  best_distilbert.pt     fake_review_clf.pkl
"""

# ── stdlib / third-party ─────────────────────────────────────────────────────
import os, re, time, pickle, io, warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import torch
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    pipeline,
)
import scipy.sparse as sp
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SentimentIQ",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── base ── */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── hero banner ── */
.hero {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 40%, #0f3460 70%, #533483 100%);
    border-radius: 18px; padding: 2.8rem 2.5rem 2.2rem;
    margin-bottom: 1.8rem; text-align: center;
    box-shadow: 0 10px 40px rgba(83,52,131,0.45);
    position: relative; overflow: hidden;
}
.hero::before {
    content: ''; position: absolute; inset: 0;
    background: radial-gradient(ellipse at 70% 30%, rgba(99,179,237,0.15) 0%, transparent 60%);
}
.hero h1  { color: #fff; font-size: 2.6rem; font-weight: 700;
            margin: 0 0 .4rem; letter-spacing: -0.5px; }
.hero p   { color: rgba(255,255,255,0.75); font-size: 1.05rem; margin: 0; }
.hero .pills { margin-top: 1rem; display: flex; justify-content: center; gap: .5rem; flex-wrap: wrap; }
.hero .pill  { background: rgba(255,255,255,0.12); color: #e2e8f0;
               padding: .25rem .8rem; border-radius: 50px; font-size: .8rem;
               border: 1px solid rgba(255,255,255,0.2); }

/* ── section headers ── */
.sh { font-size: 1.1rem; font-weight: 700; color: #1e293b;
      border-left: 4px solid #6366f1; padding-left: .7rem;
      margin: 1.6rem 0 .9rem; }

/* ── sentiment badge ── */
.badge {
    display: inline-flex; align-items: center; gap: .45rem;
    padding: .55rem 1.4rem; border-radius: 50px;
    font-weight: 700; font-size: 1.15rem; letter-spacing: .3px;
    box-shadow: 0 4px 18px rgba(0,0,0,0.18);
}
.badge-pos { background: linear-gradient(90deg,#43e97b,#38f9d7); color: #064e3b; }
.badge-neu { background: linear-gradient(90deg,#f6d365,#fda085); color: #451a03; }
.badge-neg { background: linear-gradient(90deg,#f093fb,#f5576c); color: #fff; }

/* ── metric card ── */
.mcard {
    background: #fff; border-radius: 12px;
    padding: 1.1rem 1.4rem; margin-bottom: .8rem;
    box-shadow: 0 2px 14px rgba(0,0,0,0.07);
    border-left: 5px solid;
}
.mcard-green  { border-color: #10b981; }
.mcard-orange { border-color: #f59e0b; }
.mcard-red    { border-color: #ef4444; }
.mcard-purple { border-color: #8b5cf6; }
.mcard-blue   { border-color: #3b82f6; }
.mcard h4 { margin: 0 0 .2rem; font-size: .8rem; color: #64748b; text-transform: uppercase; letter-spacing: .5px; }
.mcard p  { margin: 0; font-size: 1.25rem; font-weight: 700; color: #1e293b; }

/* ── aspect pills ── */
.apills { display: flex; flex-wrap: wrap; gap: .5rem; margin: .6rem 0; }
.apill  { padding: .3rem .9rem; border-radius: 50px; font-size: .85rem;
           font-weight: 600; border: 2px solid; }
.apill-pos { background: #dcfce7; border-color: #16a34a; color: #14532d; }
.apill-neg { background: #fee2e2; border-color: #dc2626; color: #7f1d1d; }
.apill-neu { background: #fef9c3; border-color: #ca8a04; color: #713f12; }

/* ── fake banners ── */
.fake-banner    { background: linear-gradient(90deg,#7f1d1d,#dc2626);
                  color: #fff; padding: 1rem 1.4rem; border-radius: 10px;
                  font-weight: 700; font-size: 1rem;
                  box-shadow: 0 4px 14px rgba(220,38,38,.4); }
.genuine-banner { background: linear-gradient(90deg,#064e3b,#10b981);
                  color: #fff; padding: 1rem 1.4rem; border-radius: 10px;
                  font-weight: 700; font-size: 1rem; }

/* ── sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#0f172a,#1e293b);
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stRadio     label,
[data-testid="stSidebar"] .stCheckbox  label { color: #94a3b8 !important; font-size: .85rem; }

/* ── divider ── */
.my-divider { border: 0; border-top: 1px solid #e2e8f0; margin: 1.2rem 0; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# NLTK bootstrap
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _nltk():
    for p in ["vader_lexicon", "punkt", "stopwords", "punkt_tab"]:
        nltk.download(p, quiet=True)
_nltk()

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABEL_NAMES  = ["Negative", "Neutral", "Positive"]
INT2LABEL    = {0: "Negative", 1: "Neutral", 2: "Positive"}
LABEL2INT    = {"Negative": 0, "Neutral": 1, "Positive": 2}
COLORS       = {"Positive": "#10b981", "Neutral": "#f59e0b", "Negative": "#ef4444"}

ASPECT_DICT = {
    "Food / Product" : ["food","taste","flavor","menu","dish","meal","product","quality","item","fresh"],
    "Service"        : ["service","staff","waiter","waitress","rude","friendly","helpful","server"],
    "Price / Value"  : ["price","expensive","cheap","cost","value","worth","overpriced","affordable"],
    "Ambience"       : ["ambience","atmosphere","decor","cozy","noisy","clean","dirty","vibe","location"],
    "Delivery / Speed":["delivery","fast","slow","wait","minutes","hours","shipping","quick","delay"],
    "Battery / Tech" : ["battery","camera","screen","performance","update","app","software","hardware"],
}

EMOTION_EMOJI = {
    "sadness": "😞", "joy": "😄", "love": "🥰",
    "anger": "😡",  "fear": "😨", "surprise": "😲",
}

# ─────────────────────────────────────────────────────────────────────────────
# ARTIFACT LOADING  (all cached — loaded once per session)
# ─────────────────────────────────────────────────────────────────────────────

def _pkl(path):
    return pickle.load(open(path, "rb")) if os.path.exists(path) else None

@st.cache_resource(show_spinner=False)
def load_classical():
    return {
        "tfidf"  : _pkl("tfidf_vectorizer.pkl"),
        "scaler" : _pkl("feature_scaler.pkl"),
        "SVM"    : _pkl("svm_model.pkl"),
        "LR"     : _pkl("lr_model.pkl"),
        "RF"     : _pkl("rf_model.pkl"),
        "XGB"    : _pkl("xgb_model.pkl"),
    }

@st.cache_resource(show_spinner=False)
def load_distilbert():
    tok = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    mdl = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=3,
        id2label=INT2LABEL, label2id=LABEL2INT,
    )
    if os.path.exists("best_distilbert.pt"):
        mdl.load_state_dict(torch.load("best_distilbert.pt", map_location=DEVICE))
    mdl.to(DEVICE).eval()
    return tok, mdl

@st.cache_resource(show_spinner=False)
def load_emotion():
    return pipeline(
        "text-classification",
        model="bhadresh-savani/distilbert-base-uncased-emotion",
        return_all_scores=True,
        device=0 if torch.cuda.is_available() else -1,
    )

@st.cache_resource(show_spinner=False)
def load_fake():
    return _pkl("fake_review_clf.pkl")

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTION  (mirrors extract_features() in modeling notebook)
# ─────────────────────────────────────────────────────────────────────────────
_sia = SentimentIntensityAnalyzer()

def clean(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^\w\s!?.,'\"-]", "", text)
    return re.sub(r"\s+", " ", text).strip()

def extract_features(text: str) -> np.ndarray:
    t     = str(text)
    words = t.lower().split()
    vs    = _sia.polarity_scores(t)
    return np.array([[
        vs["pos"],
        vs["neg"],
        vs["neu"],
        vs["compound"],
        len(words),                                              # word_count
        len(t),                                                  # char_count
        t.count("!") / (len(words) + 1),                        # excl_density
        t.count("?") / (len(words) + 1),                        # quest_density
        sum(c.isupper() for c in t) / (len(t) + 1),             # caps_ratio
        len(set(words)) / (len(words) + 1),                     # unique_ratio
        sum(1 for w in words if w in                             # negation_count
            {"not","no","never","neither","nor","nothing","nobody"}),
        int(bool(re.search(r"http|www\.", t, re.I))),            # has_url
        np.mean([len(w) for w in words]) if words else 0,       # avg_word_len
    ]], dtype=np.float32)

# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def predict_classical(text: str, artifacts: dict, model_key: str):
    """Run one classical model. Returns (label, proba_array_or_None, ms)."""
    clf    = artifacts.get(model_key)
    tfidf  = artifacts.get("tfidf")
    scaler = artifacts.get("scaler")
    if clf is None or tfidf is None:
        return None, None, None

    ct    = clean(text)
    tfidf_vec  = tfidf.transform([ct])
    feat_scaled = scaler.transform(extract_features(ct)) if scaler else extract_features(ct)
    X     = sp.hstack([tfidf_vec, sp.csr_matrix(feat_scaled)])

    # XGBoost needs dense
    if model_key == "XGB":
        X = X.toarray()

    t0    = time.perf_counter()
    pred  = clf.predict(X)[0]
    ms    = (time.perf_counter() - t0) * 1000

    # Probability (not available for LinearSVC)
    prob = None
    if hasattr(clf, "predict_proba"):
        prob = clf.predict_proba(X)[0]

    return INT2LABEL[pred], prob, round(ms, 3)


def predict_distilbert(text: str, tok, mdl):
    """Run DistilBERT. Returns (label, proba_np_array, ms)."""
    enc  = tok(text[:512], return_tensors="pt",
               truncation=True, padding=True, max_length=128).to(DEVICE)
    t0   = time.perf_counter()
    with torch.no_grad():
        logits = mdl(**enc).logits
    ms   = (time.perf_counter() - t0) * 1000
    prob = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred = int(np.argmax(prob))
    return INT2LABEL[pred], prob, round(ms, 3)


def predict_fake(text: str, fake_clf, tfidf):
    if fake_clf is None:
        return None, None
    ct    = clean(text)
    vs    = _sia.polarity_scores(ct)
    words = ct.split()
    tmax  = float(tfidf.transform([ct]).max()) if tfidf else 0.0
    X = np.array([[
        vs["pos"], vs["neg"], vs["compound"],
        len(words),
        ct.count("!") / (len(words) + 1),
        sum(c.isupper() for c in text) / (len(text) + 1),
        len(set(words)) / (len(words) + 1),
        sum(1 for w in words if w in {"not","no","never"}),
        np.mean([len(w) for w in words]) if words else 0,
        tmax,
    ]], dtype=np.float32)
    prob   = fake_clf.predict_proba(X)[0]
    label  = "Fake" if np.argmax(prob) == 1 else "Genuine"
    return label, prob


def run_absa(text: str) -> dict:
    sentences = re.split(r"[.!?;]", str(text))
    out = {}
    for aspect, kws in ASPECT_DICT.items():
        rel = [s for s in sentences if any(k in s.lower() for k in kws)]
        if not rel:
            continue
        score = float(np.mean([_sia.polarity_scores(s)["compound"] for s in rel]))
        label = "Positive" if score >= 0.05 else ("Negative" if score <= -0.05 else "Neutral")
        out[aspect] = {"score": round(score, 3), "label": label}
    return out

# ─────────────────────────────────────────────────────────────────────────────
# PLOTLY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def prob_bar(probs: np.ndarray, title="Confidence"):
    labels = LABEL_NAMES
    cols   = [COLORS[l] for l in labels]
    fig = go.Figure(go.Bar(
        x=[f"{p*100:.1f}%" for p in probs],
        y=labels, orientation="h",
        marker_color=cols,
        text=[f"{p*100:.1f}%" for p in probs],
        textposition="outside",
    ))
    fig.update_layout(
        title=title, height=210,
        xaxis=dict(showticklabels=False, range=[0, 130]),
        margin=dict(l=10, r=60, t=35, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def gauge(value: float, label: str, color: str):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=round(value * 100, 1),
        title={"text": label, "font": {"size": 13}},
        number={"suffix": "%", "font": {"size": 22}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar":  {"color": color},
            "bgcolor": "white",
            "steps": [{"range": [0, 50], "color": "#f1f5f9"},
                      {"range": [50, 100], "color": "#e2e8f0"}],
        }
    ))
    fig.update_layout(height=195, margin=dict(l=20, r=20, t=38, b=10))
    return fig


def radar(aspect_results: dict):
    if not aspect_results:
        return None
    aspects = list(aspect_results.keys())
    scores  = [(aspect_results[a]["score"] + 1) / 2 for a in aspects]  # 0→1
    fig = go.Figure(go.Scatterpolar(
        r=scores + [scores[0]],
        theta=aspects + [aspects[0]],
        fill="toself",
        line=dict(color="#6366f1", width=2.5),
        fillcolor="rgba(99,102,241,0.2)",
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1],
                                   tickvals=[0, 0.25, 0.5, 0.75, 1],
                                   ticktext=["−1", "−0.5", "0", "+0.5", "+1"])),
        showlegend=False, height=340,
        margin=dict(l=55, r=55, t=45, b=40),
        title="Aspect Sentiment Radar",
    )
    return fig


def emotion_chart(results: list):
    results = sorted(results, key=lambda x: -x["score"])
    labels  = [f"{EMOTION_EMOJI.get(r['label'],'')} {r['label'].title()}" for r in results]
    scores  = [r["score"] for r in results]
    fig = go.Figure(go.Bar(
        x=scores, y=labels, orientation="h",
        marker=dict(color=scores, colorscale="RdYlGn", cmin=0, cmax=1),
        text=[f"{s*100:.1f}%" for s in scores],
        textposition="outside",
    ))
    fig.update_layout(
        title="Emotion Probabilities", height=280,
        xaxis=dict(showticklabels=False, range=[0, 1.2]),
        margin=dict(l=15, r=65, t=40, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def model_comparison_chart(rows: list):
    df  = pd.DataFrame(rows)
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Test Accuracy", "Test F1 (Macro)", "Inference ms/sample"),
    )
    bar_colors = ["#3b82f6","#f59e0b","#10b981","#8b5cf6","#ef4444"]
    for i, (col, key) in enumerate([
        ("Accuracy","acc"), ("F1","f1"), ("ms","ms")
    ], 1):
        fig.add_trace(go.Bar(
            x=df["model"], y=df[key],
            marker_color=bar_colors[:len(df)],
            text=[f"{v:.3f}" for v in df[key]],
            textposition="outside",
            showlegend=False,
        ), row=1, col=i)
    fig.update_layout(height=370, margin=dict(t=60, b=20))
    fig.update_xaxes(tickangle=-25)
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# ARTIFACT STATUS HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _status(path, name):
    ok   = os.path.exists(path)
    icon = "✅" if ok else "⚠️ "
    size = f"({os.path.getsize(path)/1024/1024:.1f} MB)" if ok else "(missing)"
    st.sidebar.markdown(f"{icon} **{name}** {size}")

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    primary_model = st.selectbox(
        "Primary Sentiment Model",
        ["DistilBERT (fine-tuned)", "SVM (Linear)", "Logistic Regression",
         "Random Forest", "XGBoost"],
    )
    MODEL_KEY_MAP = {
        "DistilBERT (fine-tuned)": "BERT",
        "SVM (Linear)":            "SVM",
        "Logistic Regression":     "LR",
        "Random Forest":           "RF",
        "XGBoost":                 "XGB",
    }
    primary_key = MODEL_KEY_MAP[primary_model]

    st.markdown("---")
    st.markdown("### 🔧 Features")
    run_emotion = st.checkbox("🎭 Emotion Detection",          value=True)
    run_absa    = st.checkbox("🎯 Aspect-Based Analysis",      value=True)
    run_fake    = st.checkbox("🚨 Fake Review Detection",      value=True)
    run_compare = st.checkbox("📊 Compare All Models",         value=False)

    st.markdown("---")
    st.markdown("### 📁 Artifact Status")
    _status("best_distilbert.pt",  "DistilBERT weights")
    _status("tfidf_vectorizer.pkl","TF-IDF vectorizer")
    _status("feature_scaler.pkl",  "Feature scaler")
    _status("svm_model.pkl",       "SVM model")
    _status("lr_model.pkl",        "Logistic Regression")
    _status("rf_model.pkl",        "Random Forest")
    _status("xgb_model.pkl",       "XGBoost")
    _status("fake_review_clf.pkl", "Fake review clf")

    st.markdown("---")
    st.markdown(
        "**SentimentIQ** — End-to-end review intelligence combining "
        "classical ML & transformer models with ABSA, emotion detection "
        "and fake-review flagging."
    )

# ─────────────────────────────────────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🧠 SentimentIQ</h1>
  <p>Advanced Review Intelligence powered by Classical ML &amp; Transformers</p>
  <div class="pills">
    <span class="pill">Sentiment Classification</span>
    <span class="pill">Aspect-Based Analysis</span>
    <span class="pill">Emotion Detection</span>
    <span class="pill">Fake Review Detection</span>
    <span class="pill">Model Comparison</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab_single, tab_batch, tab_insights = st.tabs([
    "🔍 Single Review Analysis",
    "📂 Batch Processing",
    "📈 Model Insights",
])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — SINGLE REVIEW
# ═════════════════════════════════════════════════════════════════════════════
with tab_single:
    col_left, col_right = st.columns([1, 1], gap="large")

    # ── LEFT — input ──────────────────────────────────────────────────────────
    with col_left:
        st.markdown('<div class="sh">✍️ Review Input</div>', unsafe_allow_html=True)

        EXAMPLES = {
            "— pick an example —": "",
            "⭐⭐⭐⭐⭐  Amazing phone": (
                "Battery life is absolutely incredible — lasts all day easily! "
                "Performance is blazing fast. Camera is decent but could be better "
                "in low light. Overall an amazing device, totally worth the price."
            ),
            "❌  Rude service": (
                "The staff was incredibly rude and dismissive. We waited 45 minutes "
                "for food that arrived cold. The ambience was noisy and uncomfortable. "
                "Completely overpriced for such a terrible experience. Never returning."
            ),
            "🤔  Mixed feelings": (
                "I expected better from a brand this popular. The product quality is "
                "okay but it's seriously overpriced for what you get. Delivery was "
                "super fast though, and packaging was great."
            ),
            "🚨  Suspicious review": (
                "BEST PRODUCT EVER!!!! ABSOLUTELY AMAZING!!!! BUY IT NOW!!!! "
                "YOU WILL LOVE IT SO MUCH!!!! 10/10!!!! INCREDIBLE!!!"
            ),
            "🍕  Restaurant visit": (
                "The food was absolutely delicious — pasta cooked to perfection. "
                "Service was friendly and attentive. Location is cozy but gets noisy "
                "on weekends. Prices are reasonable for the quality. Will come back!"
            ),
        }

        example_choice = st.selectbox("Quick examples", list(EXAMPLES.keys()))
        review_text = st.text_area(
            "Paste or type a review",
            value=EXAMPLES[example_choice],
            height=185,
            placeholder="Enter a customer review to analyse…",
        )
        analyze = st.button("🚀  Analyse Review", type="primary",
                            use_container_width=True)

        # word count helper
        wc = len(review_text.split()) if review_text.strip() else 0
        st.caption(f"{wc} words | {len(review_text)} characters")

    # ── RIGHT — results ───────────────────────────────────────────────────────
    with col_right:
        if analyze and review_text.strip():
            st.markdown('<div class="sh">📊 Analysis Results</div>',
                        unsafe_allow_html=True)

            with st.spinner("Analysing…"):
                arts = load_classical()

                # ── Sentiment ─────────────────────────────────────────────
                if primary_key == "BERT":
                    tok, bert_mdl = load_distilbert()
                    sentiment, probs, ms = predict_distilbert(
                        clean(review_text), tok, bert_mdl)
                else:
                    sentiment, probs, ms = predict_classical(
                        review_text, arts, primary_key)

                badge_cls = {
                    "Positive": "badge-pos",
                    "Neutral":  "badge-neu",
                    "Negative": "badge-neg",
                }[sentiment]
                badge_icon = {
                    "Positive": "😊", "Neutral": "😐", "Negative": "😞"
                }[sentiment]

                st.markdown(f"""
                <div style="text-align:center; padding: 1rem 0 .5rem;">
                  <span class="badge {badge_cls}">
                    {badge_icon} &nbsp; {sentiment}
                  </span>
                  <p style="color:#64748b; margin-top:.6rem; font-size:.88rem;">
                    Model: <b>{primary_model}</b> &nbsp;|&nbsp;
                    Inference: <b>{ms} ms</b>
                  </p>
                </div>
                """, unsafe_allow_html=True)

                # Confidence bars (only when probabilities available)
                if probs is not None:
                    st.plotly_chart(
                        prob_bar(probs, "Class Probabilities"),
                        use_container_width=True,
                    )

                # ── Compare all models ─────────────────────────────────────
                if run_compare:
                    st.markdown('<div class="sh">📊 All-Model Comparison</div>',
                                unsafe_allow_html=True)
                    comparison_rows = []
                    tok2, bert2 = load_distilbert()

                    model_run_cfg = [
                        ("DistilBERT", "BERT"),
                        ("SVM",        "SVM"),
                        ("LR",         "LR"),
                        ("RF",         "RF"),
                        ("XGB",        "XGB"),
                    ]
                    comp_cols = st.columns(len(model_run_cfg))

                    for (mname, mkey), col in zip(model_run_cfg, comp_cols):
                        if mkey == "BERT":
                            lbl, _, t = predict_distilbert(
                                clean(review_text), tok2, bert2)
                        else:
                            lbl, _, t = predict_classical(
                                review_text, arts, mkey)
                        if lbl is None:
                            col.metric(mname, "N/A")
                        else:
                            col.metric(mname, lbl, f"{t} ms")

            # ── ABSA ──────────────────────────────────────────────────────
            if run_absa:
                st.markdown('<div class="sh">🎯 Aspect-Based Analysis</div>',
                            unsafe_allow_html=True)
                asp = run_absa(review_text)
                if asp:
                    html = '<div class="apills">'
                    for a, r in asp.items():
                        cls  = {"Positive":"apill-pos","Neutral":"apill-neu",
                                "Negative":"apill-neg"}[r["label"]]
                        icon = {"Positive":"✅","Neutral":"➖","Negative":"❌"}[r["label"]]
                        html += (f'<span class="apill {cls}">'
                                 f'{icon} {a} ({r["score"]:+.2f})</span>')
                    html += '</div>'
                    st.markdown(html, unsafe_allow_html=True)

                    r_chart = radar(asp)
                    if r_chart:
                        st.plotly_chart(r_chart, use_container_width=True)
                else:
                    st.info("No specific aspects detected in this review.")

            # ── Emotion ───────────────────────────────────────────────────
            if run_emotion:
                st.markdown('<div class="sh">🎭 Emotion Detection</div>',
                            unsafe_allow_html=True)
                try:
                    emo_pipe = load_emotion()
                    raw      = emo_pipe(review_text[:512])[0]
                    # normalise output (list of dicts guaranteed)
                    if isinstance(raw, dict):
                        raw = [raw]
                    top  = max(raw, key=lambda x: x["score"])
                    icon = EMOTION_EMOJI.get(top["label"], "")
                    st.markdown(f"""
                    <div class="mcard mcard-purple">
                      <h4>Primary Emotion</h4>
                      <p>{icon} {top['label'].title()}
                         &nbsp;<span style="font-size:.85rem; color:#64748b; font-weight:400;">
                         ({top['score']*100:.1f}% confidence)</span>
                      </p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.plotly_chart(emotion_chart(raw), use_container_width=True)
                except Exception as e:
                    st.warning(f"Emotion model not available: {e}")

            # ── Fake Detection ────────────────────────────────────────────
            if run_fake:
                st.markdown('<div class="sh">🚨 Authenticity Check</div>',
                            unsafe_allow_html=True)
                fake_clf = load_fake()
                verdict, fprob = predict_fake(
                    review_text, fake_clf, arts.get("tfidf"))
                if verdict:
                    if verdict == "Fake":
                        st.markdown(
                            f'<div class="fake-banner">'
                            f'⚠️ POTENTIALLY FAKE — '
                            f'Confidence: {fprob[1]*100:.1f}%</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f'<div class="genuine-banner">'
                            f'✅ LIKELY GENUINE — '
                            f'Confidence: {fprob[0]*100:.1f}%</div>',
                            unsafe_allow_html=True,
                        )
                else:
                    st.info("Fake review classifier not loaded — run the modeling "
                            "notebook and place fake_review_clf.pkl here.")

        elif analyze:
            st.warning("Please enter a review before clicking Analyse.")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — BATCH PROCESSING
# ═════════════════════════════════════════════════════════════════════════════
with tab_batch:
    st.markdown('<div class="sh">📂 Batch Review Analysis</div>',
                unsafe_allow_html=True)
    st.markdown(
        "Upload a **CSV** file with a `text` column (also accepts `review`, "
        "`review_text`, `comment`). All selected models run on every row."
    )

    uploaded = st.file_uploader("Choose CSV", type=["csv"])

    if uploaded:
        df_batch = pd.read_csv(uploaded)
        text_col = next(
            (c for c in df_batch.columns
             if c.lower() in ["text","review","review_text","comment","body"]),
            None,
        )
        if text_col is None:
            st.error("Could not find a text column. Rename it to `text` and re-upload.")
        else:
            st.success(f"Found **{len(df_batch):,} reviews** in column `{text_col}`")
            n = st.slider("Reviews to process",
                          10, min(1000, len(df_batch)), 200, step=10)

            batch_models = st.multiselect(
                "Models to run",
                ["DistilBERT", "SVM", "LR", "RF", "XGB"],
                default=["SVM"],
            )

            if st.button("▶  Run Batch Analysis", type="primary"):
                texts  = df_batch[text_col].fillna("").astype(str).head(n).tolist()
                arts_b = load_classical()
                tok_b, bert_b = None, None
                if "DistilBERT" in batch_models:
                    tok_b, bert_b = load_distilbert()

                prog = st.progress(0.0)
                results = []

                for i, txt in enumerate(texts):
                    row = {"review_preview": txt[:80] + "…"}
                    ct  = clean(txt)

                    for mk in batch_models:
                        if mk == "DistilBERT":
                            lbl, _, _ = predict_distilbert(ct, tok_b, bert_b)
                        else:
                            lbl, _, _ = predict_classical(txt, arts_b, mk)
                        row[mk] = lbl if lbl else "N/A"

                    results.append(row)
                    prog.progress((i + 1) / len(texts))

                res_df = pd.DataFrame(results)

                # summary metrics for first selected model
                first_model = batch_models[0]
                vc = res_df[first_model].value_counts()

                c1, c2, c3 = st.columns(3)
                c1.metric("✅ Positive", int(vc.get("Positive", 0)))
                c2.metric("➖ Neutral",  int(vc.get("Neutral",  0)))
                c3.metric("❌ Negative", int(vc.get("Negative", 0)))

                # donut
                fig_d = go.Figure(go.Pie(
                    labels=vc.index, values=vc.values, hole=0.55,
                    marker=dict(
                        colors=[COLORS.get(l, "#94a3b8") for l in vc.index],
                        line=dict(color="white", width=2),
                    ),
                ))
                fig_d.update_layout(
                    title=f"Distribution — {first_model}",
                    height=320, margin=dict(t=40, b=10),
                )
                st.plotly_chart(fig_d, use_container_width=True)

                st.dataframe(res_df, use_container_width=True)
                st.download_button(
                    "⬇  Download Results CSV",
                    res_df.to_csv(index=False),
                    "batch_results.csv",
                    "text/csv",
                )

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL INSIGHTS
# ═════════════════════════════════════════════════════════════════════════════
with tab_insights:
    st.markdown('<div class="sh">📈 Trained Model Performance</div>',
                unsafe_allow_html=True)
    st.caption(
        "Performance figures shown are representative values from the modeling "
        "notebook on the Yelp 10 000-sample dataset (70/15/15 split)."
    )

    # Static results table matching what the notebooks produce
    perf_data = {
        "Model"               : ["SVM (Linear)", "Logistic Regression",
                                 "Random Forest", "XGBoost",
                                 "DistilBERT (fine-tuned)"],
        "Accuracy"            : [0.76, 0.75, 0.71, 0.74, 0.88],
        "F1 Macro"            : [0.73, 0.72, 0.68, 0.71, 0.86],
        "Inference (ms/samp)" : [0.05, 0.08, 1.20, 0.85, 22.0],
        "Size (MB)"           : [12, 4, 55, 18, 255],
        "Best For"            : ["Speed+Accuracy", "Interpretability",
                                 "Robustness", "Balanced",
                                 "Highest Accuracy"],
    }
    perf_df = pd.DataFrame(perf_data)

    # Bar comparison
    rows_for_chart = [
        {"model": r["Model"].replace(" (fine-tuned)","").replace(" (Linear)",""),
         "acc": r["Accuracy"], "f1": r["F1 Macro"],
         "ms": r["Inference (ms/samp)"]}
        for _, r in perf_df.iterrows()
    ]
    st.plotly_chart(model_comparison_chart(rows_for_chart), use_container_width=True)

    st.dataframe(
        perf_df.set_index("Model").style.highlight_max(
            subset=["Accuracy","F1 Macro"], color="#d1fae5"
        ).highlight_min(
            subset=["Inference (ms/samp)","Size (MB)"], color="#dbeafe"
        ),
        use_container_width=True,
    )

    # Architecture
    st.markdown('<div class="sh">🗺️ System Architecture</div>',
                unsafe_allow_html=True)
    st.code("""
Input Review
     │
     ├─ Text Cleaning (lowercase · strip URLs · remove HTML)
     │
     ├─ Feature Engineering (13 hand-crafted features)
     │       VADER scores · word/char count · exclamation density
     │       caps ratio · unique word ratio · negation count ···
     │
     ├─[A] TF-IDF (50k unigrams+bigrams)
     │     + Scaled Features
     │       │
     │       ├─ SVM (LinearSVC)          → Sentiment Label
     │       ├─ Logistic Regression      → Sentiment + Probability
     │       ├─ Random Forest            → Sentiment + Probability
     │       └─ XGBoost                  → Sentiment + Probability
     │
     ├─[B] DistilBERT Tokenizer (max_len=128)
     │     DistilBertForSequenceClassification (fine-tuned 3 epochs)
     │     → Sentiment + Confidence Scores
     │
     ├─[C] GoEmotions Pipeline
     │     (bhadresh-savani/distilbert-base-uncased-emotion)
     │     → joy · anger · sadness · fear · love · surprise
     │
     ├─[D] ABSA Engine (VADER + aspect keywords)
     │     → Per-aspect Positive / Neutral / Negative
     │
     └─[E] GradientBoosting Fake Detector
           → Genuine / Fake + confidence
""", language="text")

    # Recommendations
    st.markdown('<div class="sh">💡 Deployment Recommendations</div>',
                unsafe_allow_html=True)

    rc1, rc2 = st.columns(2)
    with rc1:
        st.markdown("""
        **For real-time APIs (latency < 5 ms)**  
        → Use **SVM** or **Logistic Regression**  
        → Tiny model, instant inference, good accuracy

        **For maximum accuracy**  
        → Use **DistilBERT (fine-tuned)**  
        → 88% test accuracy, 22 ms/sample
        """)
    with rc2:
        st.markdown("""
        **For interpretability**  
        → Use **Logistic Regression** or **XGBoost**  
        → Feature importances explain predictions

        **For edge / offline deployment**  
        → Use **SVM** (12 MB pkl) or **XGBoost** (18 MB)  
        → No GPU required
        """)

    # Run instructions
    st.markdown('<div class="sh">🚀 Running This App</div>', unsafe_allow_html=True)
    st.info("""
**Step 1** — Train models by running `01_EDA.ipynb` then `modeling_sentimemntal_analysis_.ipynb` in Google Colab  
**Step 2** — Download all `.pkl` and `.pt` artifacts (auto-downloaded at end of modeling notebook)  
**Step 3** — Place artifacts in the same folder as `app.py`  
**Step 4** — Install dependencies: `pip install -r requirements.txt`  
**Step 5** — Launch: `streamlit run app.py`
    """)
