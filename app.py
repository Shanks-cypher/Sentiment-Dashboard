import streamlit as st
import pandas as pd
import numpy as np
import re
from io import BytesIO
from PIL import Image
from wordcloud import WordCloud, STOPWORDS
from transformers import pipeline
import torch
import plotly.express as px
import tempfile
import os

# --- SESSION STATE INITIALIZATION (FIX: Persists data across reruns) ---
if 'data_df' not in st.session_state:
    st.session_state.data_df = None

if 'chosen_col' not in st.session_state:
    st.session_state.chosen_col = None

st.set_page_config(page_title="Sentiment Dashboard (Improved)", layout="wide", initial_sidebar_state="expanded")

# -----------------------
# Helpers
# -----------------------
@st.cache_data
def load_sample_data():
    """Loads a small DataFrame for demonstration purposes."""
    return pd.DataFrame({
        "text": [
            "I love this product! It's amazing.",
            "This is the worst experience I have ever had.",
            "The service was okay, nothing special.",
            "Absolutely fantastic! Highly recommend it.",
            "I'm not happy with this purchase.",
            "Pretty good, but could be better.",
            "This is terrible, I want a refund.",
            "Great quality and fast delivery!",
            "Not bad, but not great either.",
            "Best experience ever, I am very satisfied!"
        ]
    })

def clean_text(s: str) -> str:
    """Basic cleaning: remove urls, mentions and extra whitespace."""
    if not isinstance(s, str):
        return ""
    s = re.sub(r"http\S+|www\S+|https\S+", "", s, flags=re.MULTILINE)
    s = re.sub(r"\@\w+|\#", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

@st.cache_resource(show_spinner=False)
def get_pipeline(model_name: str, device: int):
    """Create and cache the HF pipeline for a given model and device."""
    try:
        pipe = pipeline("sentiment-analysis", model=model_name, device=device)
    except Exception as e:
        st.error(f"Failed to load model {model_name}. Falling back to default: {e}")
        pipe = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=device)
    return pipe

def batch_predict(pipe, texts, batch_size=32):
    """Run predictions in batches. Returns list of dicts."""
    results = []
    n = len(texts)
    for i in range(0, n, batch_size):
        batch = texts[i : i + batch_size]
        preds = pipe(batch)
        if isinstance(preds, dict):
            preds = [preds]
        results.extend(preds)
    return results

def preds_to_df(texts, preds, label_mapping=None):
    """Convert HF preds to dataframe; apply label mapping if provided."""
    labels = []
    scores = []
    for p in preds:
        label = p.get("label") if isinstance(p, dict) else None
        score = p.get("score") if isinstance(p, dict) else None
        if label_mapping and label in label_mapping:
            mapped = label_mapping[label]
        else:
            mapped = label
        labels.append(mapped)
        scores.append(score)
    return pd.DataFrame({"text": texts, "label": labels, "score": scores})

def make_wordcloud(texts, max_words=100):
    if not texts:
        return None
    stopwords = set(STOPWORDS)
    blob = " ".join(texts)
    wc = WordCloud(width=800, height=400, collocations=False, stopwords=stopwords, max_words=max_words).generate(blob)
    return wc.to_image()

def df_to_csv_bytes(df: pd.DataFrame):
    return df.to_csv(index=False).encode("utf-8")

# -----------------------
# UI: Sidebar & Model Loading
# -----------------------
st.sidebar.title("Options")
st.sidebar.markdown("Configure model & inference settings")

MODEL_CHOICES = {
    "DistilBERT (SST-2) — fast": "distilbert-base-uncased-finetuned-sst-2-english",
    "RoBERTa (Twitter sentiment) — balanced (3 labels)": "cardiffnlp/twitter-roberta-base-sentiment",
    "BERT (multilingual rating) — 5-star style": "nlptown/bert-base-multilingual-uncased-sentiment"
}

model_choice_label = st.sidebar.selectbox("Model", list(MODEL_CHOICES.keys()), index=0)
model_name = MODEL_CHOICES[model_choice_label]

use_gpu = st.sidebar.checkbox("Use GPU (if available)", value=False)
device = 0 if (use_gpu and torch.cuda.is_available()) else -1

batch_size = st.sidebar.slider("Batch size", min_value=1, max_value=128, value=32)
show_wordclouds = st.sidebar.checkbox("Show wordclouds", value=True)
st.sidebar.markdown("---")

with st.spinner("Loading model..."):
    pipe = get_pipeline(model_name, device)

LABEL_MAPPINGS = {
    "distilbert-base-uncased-finetuned-sst-2-english": {"POSITIVE": "POSITIVE", "NEGATIVE": "NEGATIVE"},
    "cardiffnlp/twitter-roberta-base-sentiment": {
        "LABEL_0": "NEGATIVE",
        "LABEL_1": "NEUTRAL",
        "LABEL_2": "POSITIVE",
        "negative": "NEGATIVE",
        "neutral": "NEUTRAL",
        "positive": "POSITIVE"
    },
    "nlptown/bert-base-multilingual-uncased-sentiment": {
        "1 star": "NEGATIVE",
        "2 stars": "NEGATIVE",
        "3 stars": "NEUTRAL",
        "4 stars": "POSITIVE",
        "5 stars": "POSITIVE"
    }
}

label_map = LABEL_MAPPINGS.get(model_name, None)

# -----------------------
# UI: Main layout
# -----------------------
st.title("Sentiment Analysis Dashboard")
st.markdown("Paste a sentence or upload a CSV with a text column. Choose model and inference options from the sidebar.")

col1, col2 = st.columns([1, 1])

# Single text analysis
with col1:
    st.header("Single text analysis")
    single_text = st.text_area("Enter text here", height=150)
    if st.button("Analyze text"):
        if not single_text.strip():
            st.warning("Please enter some text to analyze.")
        else:
            cleaned = clean_text(single_text)[:1000]
            with st.spinner("Running inference..."):
                pred = pipe(cleaned)
            
            if isinstance(pred, list) and len(pred) > 0:
                p = pred[0]
            elif isinstance(pred, dict):
                p = pred
            else:
                p = {"label": str(pred), "score": None}
            raw_label = p.get("label")
            score = p.get("score")
            mapped_label = label_map.get(raw_label, raw_label) if label_map else raw_label
            st.subheader(f"Label: {mapped_label}")
            if score is not None:
                st.write(f"Confidence: {score:.3f}")
            st.write("**Cleaned input:**")
            st.write(cleaned)

# Batch upload
with col2:
    st.header("Batch upload (CSV)")
    
    # Use key for the uploader
    uploaded = st.file_uploader("Upload CSV file (text column)", type=["csv"], key='file_uploader')
    
    # Logic for sample data button
    if st.button("Use sample data", key='load_sample_data'):
        # Store data in session state
        st.session_state.data_df = load_sample_data()
        st.success("Sample data loaded.")
        st.rerun() # Force rerun to execute the rest of the script and show the selection box

    # Handle file upload persistence
    if uploaded is not None:
        try:
            # Check if data_df is NOT set or if the uploaded file is new
            uploaded_id = uploaded.file_id if hasattr(uploaded, 'file_id') else uploaded.name
            if st.session_state.data_df is None or uploaded_id != st.session_state.get('last_uploaded_id_key'):
                st.session_state.data_df = pd.read_csv(uploaded)
                st.session_state.last_uploaded_id_key = uploaded_id
                st.session_state.chosen_col = None # Reset column selection
                st.success("CSV loaded.")
                st.rerun() # Rerun to display selection box
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            st.session_state.data_df = None

    # Retrieve persistent data from session state
    df = st.session_state.data_df

    if df is not None:
        st.write("Preview of dataset")
        st.dataframe(df.head(10))
        
        # Detect text columns
        text_cols = [c for c in df.columns if df[c].dtype == object]
        
        if not text_cols:
            st.error("No text-like column found. Ensure your CSV has a column with textual data.")
        else:
            default = "text" if "text" in text_cols else text_cols[0]
            
            # Use persisted column choice, default otherwise
            if st.session_state.chosen_col is None or st.session_state.chosen_col not in text_cols:
                st.session_state.chosen_col = default

            chosen = st.selectbox(
                "Select text column", 
                text_cols, 
                index=text_cols.index(st.session_state.chosen_col) if st.session_state.chosen_col in text_cols else 0, 
                key='column_selector'
            )
            
            # Update the persisted column choice
            st.session_state.chosen_col = chosen

            texts = df[chosen].astype(str).apply(clean_text).tolist()
            
            # This final button will now execute because 'df' is persisted in session state
            if st.button("Run batch sentiment", key='run_analysis'):
                with st.spinner("Running batch inference..."):
                    preds = batch_predict(pipe, texts, batch_size=batch_size)
                
                out_df = preds_to_df(texts, preds, label_mapping=label_map)
                
                # Merge results
                merged = df.copy()
                merged = merged.reset_index(drop=True)
                merged["sentiment_label"] = out_df["label"]
                merged["sentiment_score"] = out_df["score"]
                
                st.success("Done!")
                st.dataframe(merged.head(50))

                # Visuals
                st.subheader("Label distribution")
                dist = merged["sentiment_label"].fillna("UNKNOWN").value_counts().reset_index()
                dist.columns = ["label", "count"]
                fig = px.bar(dist, x="label", y="count", title="Sentiment distribution")
                st.plotly_chart(fig, use_container_width=True)

                # Wordclouds
                if show_wordclouds:
                    pos_texts = merged[merged["sentiment_label"] == "POSITIVE"][chosen].astype(str).tolist()
                    neg_texts = merged[merged["sentiment_label"] == "NEGATIVE"][chosen].astype(str).tolist()
                    colp, coln = st.columns(2)
                    with colp:
                        st.write("### Positive wordcloud")
                        img = make_wordcloud(pos_texts)
                        # FIX: Changed use_column_width=True to use_container_width=True
                        if img:
                            st.image(img, use_container_width=True) 
                        else:
                            st.write("No positive texts available.")
                    with coln:
                        st.write("### Negative wordcloud")
                        img2 = make_wordcloud(neg_texts)
                        # FIX: Changed use_column_width=True to use_container_width=True
                        if img2:
                            st.image(img2, use_container_width=True)
                        else:
                            st.write("No negative texts available.")

                # Download
                csv_bytes = df_to_csv_bytes(merged)
                st.download_button("Download predictions (CSV)", csv_bytes, file_name="sentiment_predictions.csv", mime="text/csv")

# Footer
st.markdown("---")
st.markdown("Built for improved model selection, caching, batching, and cleaner visuals. Recommended to run on a machine with sufficient memory for larger models.")