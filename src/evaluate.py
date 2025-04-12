from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Used during training
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted")
    }

# Save classification report
def save_metrics(y_true, y_pred, label_names, out_path="outputs/metrics/report.json"):
    report = classification_report(y_true, y_pred, target_names=label_names, output_dict=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=4)

# Save confusion matrix as image
def save_confusion_matrix(y_true, y_pred, label_names, out_path="outputs/metrics/confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=label_names, yticklabels=label_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_path)

