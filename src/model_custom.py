from transformers import AutoTokenizer, AutoModelForSequenceClassification

def load_model():
    model_path = "../outputs/model/"
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return model

def load_tokenizer():
    model_path = "../outputs/model/"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return tokenizer

