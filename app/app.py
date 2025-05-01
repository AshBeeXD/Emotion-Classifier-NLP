import streamlit as st
st.set_page_config(page_title="Emotion Classifier", layout="centered")  # MUST BE FIRST

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

# Load GoEmotions label names
GOEMOTIONS_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization", "relief",
    "remorse", "sadness", "surprise", "neutral"
]

# Load model and tokenizer
MODEL_PATH = "outputs/model"
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# Streamlit UI
st.title("🧠 Emotion Classifier (NLP)")
st.markdown("Enter a sentence to analyze:")

input_text = st.text_area(" ", height=100)

if st.button("Classify") and input_text.strip():
    with torch.no_grad():
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
        pred_label_idx = torch.argmax(probs).item()
        pred_emotion = GOEMOTIONS_LABELS[pred_label_idx]

        st.success(f"**Predicted Emotion:** {pred_emotion}")

        st.markdown("### Prediction Probabilities:")
        for i, prob in enumerate(probs):
            st.write(f"- {GOEMOTIONS_LABELS[i]}: {prob.item():.4f}")

