# Emotion Classifier (NLP)

A simple NLP-based emotion classification app that uses a fine-tuned transformer model on the GoEmotions dataset to predict the emotion conveyed in a given sentence.

## Project Structure

Emotion-Classifier-NLP/
├── notebooks/
│   ├── 01_exploration.ipynb
│   ├── 02_training.ipynb
│   ├── 03_evaluation.ipynb
│   └── 04_comparison.ipynb
├── src/
│   ├── data_loader.py
│   ├── model.py
│   ├── model_hartmann.py
│   ├── model_custom.py
│   ├── train.py
│   └── evaluate.py
├── app/
│   └── app.py
├── outputs/
│   ├── model/                  # Trained model
│   ├── metrics/                # Evaluation metrics
│   └── interpretations/        # Integrated gradients visualization plots
├── requirements.txt
├── README.md
└── .gitignore

## Features
- Uses Hugging Face Transformers with a fine-tuned model
- Classifies emotion from text across 28 GoEmotions labels
- Streamlit frontend for interactive use
- Displays model prediction probabilities
- Shows sample integrated gradients visualizations per label (optional)

## Running the App
Make sure you have Streamlit and other dependencies installed:

pip install -r requirements.txt

Then start the app:

streamlit run app/app.py

## Example Output
- Input: "I find this funny"
- Output: Predicted Emotion: `amusement`
- Shows prediction probabilities across all 28 classes

## Notes
- Pretrained model is saved in `outputs/model/`
- Integrated Gradients plots should be saved under `outputs/interpretations/` and named using the format: `sample_{n}_{label}.png`

## Credits
Based on the GoEmotions dataset by Google Research.

---

# 📊 Performance Metrics

| Metric     | Score |
|------------|-------|
| Accuracy   | 60.2% |
| Macro F1   | 48.3% |
| Weighted F1| 59.6% |

_Confusion matrix available in `outputs/metrics/confusion_matrix.png`._

---

# 🔍 Model Comparison (Hartmann vs Custom Model)

| Sample Sentence                                     | Hartmann Prediction(s)     | Custom Model Prediction(s) |
|-----------------------------------------------------|-----------------------------|-----------------------------|
| I love spending time with my family.                | joy, sadness, disgust       | love, joy, admiration       |
| This is the worst day of my life.                   | disgust, anger, sadness     | anger, surprise, disgust    |
| I'm feeling very nervous about the exam.            | fear, sadness, joy          | nervousness, fear, embarrassment |
| What a beautiful sunset!                            | joy, surprise, neutral      | admiration, excitement, joy |
| I feel so disappointed and frustrated.              | sadness, anger, disgust     | disappointment, annoyance, anger |
| I'm not sure how to feel about this.                | neutral, disgust, sadness   | confusion, optimism, disapproval |
| That was hilarious, I can't stop laughing!          | joy, surprise, neutral      | amusement, joy, optimism    |
| I feel completely empty and lost.                   | sadness, neutral, disgust   | surprise, disappointment, optimism |

**Insights**:
- The custom model captured more nuanced emotions, while Hartmann’s model tended to favor high-level emotions.
- Some variance due to differences in label granularity between the models.
- The custom model showed stronger performance in emotions like admiration, amusement, and disappointment.

---

# ⚠ Known Limitations

- **Single-label restriction**: While the data supports multi-label emotion classification, the model currently predicts only the highest probability emotion.
- **Low support for some classes**: Emotions like grief and pride had low representation in the training data.
- **Data bias**: Results reflect Reddit comment biases present in the GoEmotions dataset.

---

# 🔎 Confidence Threshold

A **confidence threshold of 0.6** is applied in the app.  
If the top emotion’s probability is below this value, the app returns:

"Unclear / Not enough signal"

This prevents overconfident predictions on uncertain or ambiguous text.

---

# 🔮 Future Work

- Expand to multi-label predictions to better capture complex emotions.
- Improve minority class performance via data augmentation or rebalancing.
- Incorporate explainability methods directly into the Streamlit app.
- Deploy the app to Streamlit Cloud or Hugging Face Spaces.
- Collect user feedback for real-world validation.

---

# 👥 Credits

- **Model architecture**: [RoBERTa](https://huggingface.co/roberta-base) (Hugging Face)
- **Training dataset**: [GoEmotions](https://huggingface.co/datasets/go_emotions)
- **Reference model**: [Hartmann et al. (2023)](https://arxiv.org/abs/2305.05894)
- **Streamlit App Framework**: [Streamlit](https://streamlit.io/)
- **Berta Emotion Model**: [BERTa](https://huggingface.co/matejklemen/berta-base-emotion)
- **Transformers Library**: [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)

---

# 📝 License

This project is licensed under the MIT License.  
You can use, modify, and distribute the software freely, but there is no warranty.

