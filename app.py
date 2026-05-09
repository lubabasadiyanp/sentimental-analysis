import os
import re
import pickle
 
import streamlit as st
 
# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="🔍",
    layout="centered",
)
 
# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500&display=swap');
 
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'DM Serif Display', serif; }
 
.stApp { background: #0f0f13; color: #e8e6df; }
 
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.6rem;
    color: #f0ece3;
    margin-bottom: 0;
    line-height: 1.15;
}
.hero-sub {
    font-size: 0.95rem;
    color: #7a7870;
    margin-top: 6px;
    margin-bottom: 32px;
    font-weight: 300;
}
.result-card {
    border-radius: 14px;
    padding: 28px 32px;
    margin-top: 24px;
    border: 1px solid rgba(255,255,255,0.07);
    background: #1a1a22;
}
.result-label {
    font-size: 0.78rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #7a7870;
    margin-bottom: 8px;
    font-weight: 500;
}
.result-sentiment {
    font-family: 'DM Serif Display', serif;
    font-size: 2.8rem;
    font-weight: 400;
    margin: 0;
}
.sentiment-positive { color: #6fcf97; }
.sentiment-negative { color: #eb5757; }
.sentiment-neutral  { color: #f2c94c; }
.confidence-bar-bg {
    background: rgba(255,255,255,0.07);
    border-radius: 99px;
    height: 6px;
    margin-top: 14px;
}
.confidence-bar-fill { height: 6px; border-radius: 99px; }
.conf-text { font-size: 0.82rem; color: #7a7870; margin-top: 6px; }
.pill-model {
    display: inline-block;
    font-size: 0.72rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 3px 10px;
    border-radius: 99px;
    background: rgba(255,255,255,0.07);
    color: #a0998f;
    margin-top: 16px;
    font-weight: 500;
}
.warning-box {
    background: rgba(242,201,76,0.08);
    border: 1px solid rgba(242,201,76,0.25);
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 0.82rem;
    color: #c9a83c;
    margin-top: 14px;
}
.stButton > button {
    width: 100%;
    background: #f0ece3;
    color: #0f0f13;
    border: none;
    border-radius: 10px;
    padding: 14px 0;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.95rem;
    font-weight: 500;
    cursor: pointer;
}
.stTextArea textarea {
    background: #1a1a22 !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: #e8e6df !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
}
hr { border-color: rgba(255,255,255,0.06); margin: 28px 0; }
</style>
""", unsafe_allow_html=True)
 
 
# ── Model loaders (cached — runs only once per session) ───────────────────────
@st.cache_resource
def load_svm():
    try:
        if os.path.exists("tfidf_vectorizer.pkl") and os.path.exists("svm_model.pkl"):
            with open("tfidf_vectorizer.pkl", "rb") as f:
                tfidf = pickle.load(f)
            with open("svm_model.pkl", "rb") as f:
                svm = pickle.load(f)
            return tfidf, svm
    except Exception as e:
        st.warning(f"SVM models could not be loaded: {e}")
    return None, None
 
 
@st.cache_resource
def load_bert():
    try:
        import torch
        from transformers import (
            DistilBertTokenizerFast,
            DistilBertForSequenceClassification,
        )
        tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
        model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=3
        )
        if os.path.exists("best_distilbert.pt"):
            model.load_state_dict(
                torch.load("best_distilbert.pt", map_location="cpu")
            )
        else:
            st.warning("best_distilbert.pt not found — using base weights (may be inaccurate).")
        model.eval()
        return tokenizer, model
    except Exception as e:
        st.warning(f"BERT model could not be loaded: {e}")
    return None, None
 
 
# ── Helpers ───────────────────────────────────────────────────────────────────
LABEL_MAP = {0: "Negative", 1: "Neutral", 2: "Positive"}
 
def clean_text(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^\w\s!?.,'\"-]", "", text)
    return re.sub(r"\s+", " ", text).strip()
 
 
def predict_svm(text, tfidf, svm):
    cleaned = clean_text(text)
    vec  = tfidf.transform([cleaned])
    pred = svm.predict(vec)[0]
    sentiment = LABEL_MAP.get(int(pred), str(pred)) if isinstance(pred, (int, float)) else str(pred)
    return sentiment, None, False
 
 
def predict_bert(text, tokenizer, model):
    import torch
    cleaned = clean_text(text)
    original_len = len(tokenizer.encode(cleaned, add_special_tokens=True))
    tokens = tokenizer(cleaned, return_tensors="pt", max_length=128,
                       truncation=True, padding="max_length")
    with torch.no_grad():
        logits = model(**tokens).logits
    pred       = logits.argmax(dim=1).item()
    confidence = torch.softmax(logits, dim=1).max().item()
    return LABEL_MAP[pred], round(confidence, 4), original_len > 128
 
 
def render_result(sentiment, confidence, model_name, truncated):
    css_class  = f"sentiment-{sentiment.lower()}"
    emoji      = {"Positive": "✦", "Negative": "✕", "Neutral": "◉"}.get(sentiment, "")
    bar_color  = {"Positive": "#6fcf97", "Negative": "#eb5757", "Neutral": "#f2c94c"}.get(sentiment, "#888")
 
    conf_html = ""
    if confidence is not None:
        pct = int(confidence * 100)
        conf_html = f"""
        <div class="confidence-bar-bg">
          <div class="confidence-bar-fill" style="width:{pct}%;background:{bar_color};"></div>
        </div>
        <div class="conf-text">Confidence: {pct}%</div>"""
 
    trunc_html = '<div class="warning-box">⚠ Input exceeded 128 tokens and was truncated.</div>' if truncated else ""
 
    st.markdown(f"""
    <div class="result-card">
      <div class="result-label">Sentiment detected</div>
      <div class="result-sentiment {css_class}">{emoji} {sentiment}</div>
      {conf_html}
      {trunc_html}
      <div class="pill-model">{model_name}</div>
    </div>""", unsafe_allow_html=True)
 
 
# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">Sentiment Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Product review analysis · DistilBERT or SVM · Negative · Neutral · Positive</div>', unsafe_allow_html=True)
 
model_choice = st.radio("Model", ["DistilBERT", "SVM"], horizontal=True, label_visibility="collapsed")
 
review_text = st.text_area(
    "Review",
    placeholder="Paste a product review here…",
    height=160,
    label_visibility="collapsed",
)
 
if st.button("Analyse sentiment"):
    if not review_text.strip():
        st.error("Please enter some text before analysing.")
    elif model_choice == "SVM":
        tfidf, svm = load_svm()
        if svm is None:
            st.error("SVM model files not found. Make sure `tfidf_vectorizer.pkl` and `svm_model.pkl` are committed to your repo.")
        else:
            with st.spinner("Analysing…"):
                sentiment, conf, truncated = predict_svm(review_text, tfidf, svm)
            render_result(sentiment, conf, "SVM · TF-IDF", truncated)
    else:
        tokenizer, bert_model = load_bert()
        if bert_model is None:
            st.error("BERT model could not be loaded. Check your requirements.txt includes torch and transformers.")
        else:
            with st.spinner("Running DistilBERT…"):
                sentiment, conf, truncated = predict_bert(review_text, tokenizer, bert_model)
            render_result(sentiment, conf, "DistilBERT", truncated)
 
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown('<div style="font-size:0.78rem;color:#4a4845;text-align:center;">3-class · Negative · Neutral · Positive</div>', unsafe_allow_html=True)
