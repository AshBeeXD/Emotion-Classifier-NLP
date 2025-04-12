from datasets import load_dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd

def load_and_prepare_data():
    # Load dataset directly from Hugging Face
    raw_dataset = load_dataset("go_emotions")

    # Convert to DataFrame for easy manipulation
    df = pd.DataFrame(raw_dataset["train"])

    # Grab only examples with a single label (simplification)
    df = df[df["labels"].apply(lambda x: len(x) == 1)].reset_index(drop=True)
    
    # Keep just the first label in list
    df["label"] = df["labels"].apply(lambda x: x[0])

    # Clean text if needed
    df["clean_text"] = df["text"].str.lower()

    # Map integer label to string label using Hugging Face's label list
    label_names = raw_dataset["train"].features["labels"].feature.names
    df["emotion"] = df["label"].apply(lambda x: label_names[x])

    # Encode emotion names for training
    label_encoder = LabelEncoder()
    df["label_encoded"] = label_encoder.fit_transform(df["emotion"])

    # Split into train/test
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df["clean_text"].tolist(),
        df["label_encoded"].tolist(),
        test_size=0.2,
        stratify=df["label_encoded"],
        random_state=42
    )

    return train_texts, test_texts, train_labels, test_labels, label_encoder

