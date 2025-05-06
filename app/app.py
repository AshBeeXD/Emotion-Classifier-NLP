import os
os.system("git lfs pull")

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

st.set_page_config(page_title="Emotion Classifier", layout="centered")

# Load GoEmotions label names
GOEMOTIONS_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization", "relief",
    "remorse", "sadness", "surprise", "neutral"
]

# -----------------------------
# Load model and tokenizer
# -----------------------------
MODEL_PATH = "outputs/model"
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, device_map="auto", low_cpu_mem_usage=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
print("Model device:", next(model.parameters()).device)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🧠 Emotion Classifier (NLP)")
st.markdown("Enter a sentence to analyze:")

input_text = st.text_area(" ", height=100)

if st.button("Classify") and input_text.strip():

    with torch.no_grad():
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]

    pred_label_idx = torch.argmax(probs).cpu().item()
    pred_score = probs[pred_label_idx].cpu().item()
    pred_emotion = GOEMOTIONS_LABELS[pred_label_idx]

    # -----------------------------
    # Confidence threshold logic
    # -----------------------------
    threshold = 0.6
    if pred_score < threshold:
        st.warning(f"**Predicted Emotion:** Unclear / Not enough signal (Confidence: {pred_score:.0%})")
    else:
        st.success(f"**Predicted Emotion:** {pred_emotion} (Confidence: {pred_score:.0%})")

    # -----------------------------
    # Show all probabilities
    # -----------------------------
    st.markdown("### Prediction Probabilities:")
    for i, prob in enumerate(probs):
        st.write(f"- {GOEMOTIONS_LABELS[i]}: {prob.item():.4f}")

