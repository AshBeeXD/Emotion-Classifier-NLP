from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load the current model from the existing folder
model = AutoModelForSequenceClassification.from_pretrained("outputs/model")
tokenizer = AutoTokenizer.from_pretrained("outputs/model")

# Save to a new folder without any GPU/device info
model.save_pretrained("outputs/model-clean")
tokenizer.save_pretrained("outputs/model-clean")

print("Model saved to outputs/model-clean")

