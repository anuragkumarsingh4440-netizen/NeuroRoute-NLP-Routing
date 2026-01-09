import streamlit as st
import numpy as np
import joblib
import os
import tensorflow as tf
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="Complaint Intelligence System", layout="centered")

st.markdown(
    """
    <style>
    body { background-color: #0e1117; color: #fafafa; }
    .block { background-color: #161b22; padding: 20px; border-radius: 12px; margin-top: 15px; }
    .stTextArea textarea { background-color: #0e1117; color: #fafafa; border-radius: 8px; }
    .stButton button { background-color: #238636; color: white; width: 100%; height: 3em; border-radius: 8px; }
    </style>
    """,
    unsafe_allow_html=True
)

LABEL_MAP = {
    "Credit card": "Credit Card Issues",
    "Credit card or prepaid card": "Credit / Prepaid Card Issues",
    "Payday loan, title loan, personal loan, or advance loan": "Loan & EMI Issues",
    "Payday loan, title loan, or personal loan": "Personal Loan Issues",
    "Student loan": "Student Loan Issues",
    "Debt collection": "Debt Collection & Recovery",
    "Checking or savings account": "Bank Account & Debit Card Issues",
    "Money transfer, virtual currency, or money service": "Money Transfer / Wallet Issues"
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "final_model")

@st.cache_resource
def load_model_and_encoder():
    model = tf.keras.models.load_model(
        os.path.join(MODEL_PATH, "final_neural_model.keras"),
        compile=False
    )
    encoder = joblib.load(os.path.join(MODEL_PATH, "label_encoder.pkl"))
    return model, encoder

@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

model, label_encoder = load_model_and_encoder()
embedder = load_embedder()

if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "last_text" not in st.session_state:
    st.session_state.last_text = ""

st.markdown(
    "<h1 style='text-align:center;'>AI-Powered Complaint Intelligence</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align:center; color:#8b949e;'>Understand complaint type, business impact, and routing direction</p>",
    unsafe_allow_html=True
)

st.markdown("<div class='block'>", unsafe_allow_html=True)

user_text = st.text_area(
    "Paste customer complaint text",
    height=160,
    placeholder="Example: My loan EMI was deducted twice and the bank is not refunding the amount."
)

if user_text != st.session_state.last_text:
    st.session_state.prediction = None
    st.session_state.last_text = user_text

if st.button("Analyze Complaint"):
    if user_text.strip():
        embedding = embedder.encode([user_text])
        st.session_state.prediction = model.predict(embedding)[0]

st.markdown("</div>", unsafe_allow_html=True)

if st.session_state.prediction is not None:
    probs = st.session_state.prediction
    idx = np.argsort(probs)[-3:][::-1]
    raw_labels = label_encoder.inverse_transform(idx)
    labels = [LABEL_MAP.get(l, l) for l in raw_labels]
    scores = probs[idx] * 100

    st.markdown("<div class='block'>", unsafe_allow_html=True)

    st.subheader("Complaint Routing Insight")

    for i in range(3):
        st.write(f"{i+1}. {labels[i]}  —  {scores[i]:.2f}%")

    st.markdown("")

    st.bar_chart(
        {"Category": labels, "Confidence (%)": scores},
        x="Category",
        y="Confidence (%)",
        height=280
    )

    risk_note = "Operational review recommended."
    if scores[0] > 60:
        risk_note = "High priority complaint. Immediate attention advised."

    st.markdown(
        f"<p style='color:#8b949e;'>Business Insight: {risk_note}</p>",
        unsafe_allow_html=True
    )

    st.markdown("</div>", unsafe_allow_html=True)
