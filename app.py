# Complaint Neuro Intelligence System
# This app analyzes customer complaints using embeddings + neural model
# It predicts complaint category, confidence, and routing insights

import streamlit as st
import numpy as np
import joblib
import os
import tensorflow as tf
from sentence_transformers import SentenceTransformer

# --- Page setup ---
# We set page title and layout for a clean centered UI
st.set_page_config(page_title="Complaint Neuro Intelligence System", layout="centered")

# --- Custom Styling ---
# Gradient background + styled blocks/buttons for classy look
st.markdown(
    """
    <style>
    body {
        background: linear-gradient(135deg, #1f1c2c, #928dab);
        color: #f0f0f0;
        font-family: 'Segoe UI', sans-serif;
    }
    .block {
        background-color: rgba(255,255,255,0.05);
        padding: 20px;
        border-radius: 15px;
        margin-top: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    .stTextArea textarea {
        background-color: #2c2c34;
        color: #f0f0f0;
        border-radius: 8px;
    }
    .stButton button {
        background: linear-gradient(90deg, #ff6a00, #ee0979);
        color: white;
        width: 100%;
        height: 3em;
        border-radius: 8px;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Label Mapping ---
# Map raw model labels to human-friendly names with emojis
LABEL_MAP = {
    "Credit card": "💳 Credit Card Issues",
    "Credit card or prepaid card": "💳 Credit / Prepaid Card Issues",
    "Payday loan, title loan, personal loan, or advance loan": "💰 Loan & EMI Issues",
    "Payday loan, title loan, or personal loan": "💰 Personal Loan Issues",
    "Student loan": "🎓 Student Loan Issues",
    "Debt collection": "📩 Debt Collection & Recovery",
    "Checking or savings account": "🏦 Bank Account & Debit Card Issues",
    "Money transfer, virtual currency, or money service": "💸 Money Transfer / Wallet Issues"
}

MODEL_PATH = "final_model"

# --- Load Model + Encoder ---
# Neural model predicts complaint type, encoder maps labels back
@st.cache_resource
def load_model_and_encoder():
    model = tf.keras.models.load_model(
        os.path.join(MODEL_PATH, "final_neural_model.keras"),
        compile=False
    )
    encoder = joblib.load(os.path.join(MODEL_PATH, "label_encoder.pkl"))
    return model, encoder

# --- Load Embedder ---
# SentenceTransformer converts text into embeddings for model input
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

model, label_encoder = load_model_and_encoder()
embedder = load_embedder()

# --- Session State ---
# Store prediction + last text to avoid re-running unnecessarily
if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "last_text" not in st.session_state:
    st.session_state.last_text = ""

# --- Header ---
# Title and subtitle for app branding
st.markdown("<h1 style='text-align:center;'>✨ Complaint Neuro Intelligence System ✨</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#ddd;'>Smart complaint routing with classy insights</p>", unsafe_allow_html=True)
st.markdown("<div class='block'>", unsafe_allow_html=True)

# --- Input Area ---
# User pastes complaint text here for analysis
user_text = st.text_area(
    "📝 Paste customer complaint text",
    height=160,
    placeholder="Example: My loan EMI was deducted twice and the bank is not refunding the amount."
)

if user_text != st.session_state.last_text:
    st.session_state.prediction = None
    st.session_state.last_text = user_text

# --- Analyze Button ---
# Spinner shows while model processes embeddings + prediction
if st.button("🔍 Analyze Complaint"):
    if user_text.strip():
        with st.spinner("⚡ Analyzing complaint... please wait"):
            embedding = embedder.encode([user_text])
            st.session_state.prediction = model.predict(embedding)[0]
        st.success("✅ Analysis complete! Results below 👇")

st.markdown("</div>", unsafe_allow_html=True)

# --- Prediction Results ---
# Show top 3 categories with confidence + bar chart + business insight
if st.session_state.prediction is not None:
    probs = st.session_state.prediction
    idx = np.argsort(probs)[-3:][::-1]
    raw_labels = label_encoder.inverse_transform(idx)
    labels = [LABEL_MAP.get(l, l) for l in raw_labels]
    scores = probs[idx] * 100

    st.markdown("<div class='block'>", unsafe_allow_html=True)
    st.subheader("📊 Complaint Routing Insight")

    for i in range(3):
        st.write(f"{i+1}. {labels[i]} — {scores[i]:.2f}%")

    st.bar_chart(
        {"Category": labels, "Confidence (%)": scores},
        x="Category",
        y="Confidence (%)",
        height=280
    )

    if scores[0] > 60:
        st.success("🔥 High priority complaint! Immediate attention advised.")
    else:
        st.info("🛠 Operational review recommended.")

    st.markdown("</div>", unsafe_allow_html=True)
