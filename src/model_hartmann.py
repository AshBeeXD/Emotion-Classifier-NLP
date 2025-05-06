from transformers import AutoTokenizer, AutoModelForSequenceClassification

def load_tokenizer():
    model_name = "j-hartmann/emotion-english-distilroberta-base"
    return AutoTokenizer.from_pretrained(model_name)

def load_model():
    model_name = "j-hartmann/emotion-english-distilroberta-base"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return model

