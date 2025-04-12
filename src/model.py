from transformers import BertTokenizer, BertForSequenceClassification

# Load tokenizer
def get_tokenizer():
    return BertTokenizer.from_pretrained("bert-base-uncased")

# Load pre-trained BERT with classification head
def get_model(num_labels):
    return BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)

