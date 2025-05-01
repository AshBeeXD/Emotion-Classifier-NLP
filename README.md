# Emotion Classifier (NLP)

A simple NLP-based emotion classification app that uses a fine-tuned transformer model on the GoEmotions dataset to predict the emotion conveyed in a given sentence.

## Project Structure

```
Emotion-Classifier-NLP/
├── notebooks/
│   ├── 01_exploration.ipynb
│   ├── 02_training.ipynb
│   └── 03_evaluation.ipynb
├── src/
│   ├── data_loader.py
│   ├── model.py
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
```

## Features
- Uses Hugging Face Transformers with a fine-tuned model
- Classifies emotion from text across 28 GoEmotions labels
- Streamlit frontend for interactive use
- Displays model prediction probabilities
- Shows sample integrated gradients visualizations per label (optional)

## Running the App
Make sure you have Streamlit and other dependencies installed:
```bash
pip install -r requirements.txt
```

Then start the app:
```bash
streamlit run app/app.py
```

## Example Output
- Input: "I love this game"
- Output: Predicted Emotion: `love`
- Shows prediction probabilities across all 28 classes

## Notes
- Pretrained model is saved in `outputs/model/`
- Integrated Gradients plots should be saved under `outputs/interpretations/` and named using the format: `sample_{n}_{label}.png`

## Credits
Based on the GoEmotions dataset by Google Research.

