# Complaint Neuro Intelligence System
# This app analyzes customer complaints using embeddings + neural model
# It predicts complaint category, confidence, and routing insights

import streamlit as st
import numpy as np
import joblib
import tensorflow as tf
import os
from sentence_transformers import SentenceTransformer


st.set_page_config(
    page_title="NeuroRoute | AI Complaint Intelligence",
    layout="centered"
)


st.markdown(
    """
    <style>
    body {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        color: #f5f5f5;
        font-family: 'Segoe UI', sans-serif;
    }
    .block {
        background: rgba(255,255,255,0.06);
        padding: 22px;
        border-radius: 16px;
        margin-top: 20px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.35);
    }
    .stTextArea textarea {
        background-color: #1e1e2f;
        color: #ffffff;
        border-radius: 10px;
        border: 1px solid #444;
    }
    .stButton button {
        background: linear-gradient(90deg, #ff512f, #dd2476);
        color: white;
        width: 100%;
        height: 3.2em;
        border-radius: 10px;
        font-weight: bold;
        border: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE_DIR, "final_neural_model.keras")
ENCODER_FILE = os.path.join(BASE_DIR, "label_encoder.pkl")


@st.cache_resource
def load_model_and_encoder():
    model = tf.keras.models.load_model(MODEL_FILE, compile=False)
    encoder = joblib.load(ENCODER_FILE)
    return model, encoder


@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")


model, label_encoder = load_model_and_encoder()
embedder = load_embedder()


LABEL_MAP = {
    "Credit card": "💳 Credit Card Issues",
    "Credit card or prepaid card": "💳 Credit / Prepaid Card",
    "Debt collection": "📩 Debt Collection",
    "Checking or savings account": "🏦 Bank Account",
    "Student loan": "🎓 Student Loan",
    "Payday loan, title loan, personal loan, or advance loan": "💰 Loan & EMI",
    "Money transfer, virtual currency, or money service": "💸 Wallet / Transfer"
}


st.markdown("<h1 style='text-align:center;'>🧠 NeuroRoute</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center; color:#dcdcdc;'>AI-Powered Complaint Intelligence & Smart Routing</p>",
    unsafe_allow_html=True
)


st.markdown("<div class='block'>", unsafe_allow_html=True)

user_text = st.text_area(
    "📝 Paste customer complaint text",
    height=160,
    placeholder="Example: My EMI was deducted twice and no refund has been issued."
)

if st.button("🚀 Analyze Complaint"):
    if user_text.strip():
        with st.spinner("⚡ Understanding complaint context..."):
            embedding = embedder.encode([user_text])
            probs = model.predict(embedding, verbose=0)[0]
        st.session_state.probs = probs
    else:
        st.warning("⚠️ Please enter a complaint text.")

st.markdown("</div>", unsafe_allow_html=True)


if "probs" in st.session_state:
    probs = st.session_state.probs
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
        height=300
    )

    if scores[0] > 60:
        st.success("🔥 High-risk complaint detected. Immediate action advised.")
    else:
        st.info("🛠 Medium-risk complaint. Operational review recommended.")

    st.markdown("</div>", unsafe_allow_html=True)
