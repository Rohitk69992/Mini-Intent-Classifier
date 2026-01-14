# Mini Intent Classification 

This project implements a simple **Intent Classification system** that predicts whether a given piece of text expresses **High**, **Medium**, or **Low** user intent. The system is designed to simulate social-media or short-text intent detection using classical machine learning techniques.

---

## Problem Statement

Given a short text input (e.g. a social media post), classify the **intent level** of the user into one of the following categories:
- **High Intent** – clear purchase or action intent  
- **Medium Intent** – exploratory or evaluative intent  
- **Low Intent** – informational or general discussion  

---

## Tech Stack Choice

The assignment allowed choosing one option from multiple alternatives.

**Chosen stack:**
- **Model:** Logistic Regression (scikit-learn)
- **API Framework:** Flask

**Reasoning:**
- Logistic Regression is fast, interpretable, and suitable for short-text classification
- Flask provides a lightweight API layer ideal for rapid prototyping
- This combination keeps the system simple, explainable, and easy to deploy

Other options mentioned in the assignment (Naive Bayes, LSTM, FastAPI) were intentionally not used, as the requirement was to choose **one stable approach**, not all.

---

## Project Structure

```text
intent-classifier/
│
├── app/
│   ├── app.py               # Flask API
│   ├── preprocess.py        # Text cleaning logic
│   ├── intent_model.py      # Model inference
│   └── templates/
│       └── index.html       # Simple UI
│
├── data/
│   └── train.csv            # Training dataset
│
├── results/
│   └── predictions.json     # Stored predictions
│
├── train_model.py           # Model training & evaluation
├── vectorizer.pkl           # Saved TF-IDF vectorizer
├── model.pkl                # Saved trained model
├── requirements.txt
├── README.md
└── NOTES.txt

Text Preprocessing
Before prediction, input text is cleaned using:

Lowercasing

Removal of punctuation and special characters

Removal of extra whitespace

Both original text and cleaned text are retained:

Original text → traceability

Cleaned text → model input

Model Training
Vectorization: TF-IDF (unigrams + bigrams)

Classifier: Logistic Regression

Class balancing enabled

Train/Test split: 75% / 25%

Model Performance
The model was evaluated using a held-out test set.

Accuracy: ~0.65 – 0.80 (varies due to small dataset size)

Performance is limited by dataset size and linguistic overlap between classes

Important Note on Confidence
Confidence represents the model’s certainty in its predicted class, not the “strength” of intent itself.

Example:

Plaintext

intent = "low"
confidence = 0.81
means the model is very sure the text is low intent.

API Endpoints
GET /
Serves a simple HTML interface for testing.

POST /predict
Accepts JSON input:

JSON

{
  "text": "I need a tool that helps me generate leads quickly"
}
Returns:

JSON

{
  "text": "I need a tool that helps me generate leads quickly",
  "clean_text": "i need a tool that helps me generate leads quickly",
  "intent": "high",
  "confidence": 0.72,
  "timestamp": "2026-01-14T12:00:00"
}
Predictions are also appended to results/predictions.json.

How to Run
Install dependencies:

Bash

pip install -r requirements.txt
Train the model:

Bash

python train_model.py
Start the API:

Bash

python app/app.py
Open in browser:

Plaintext

[http://127.0.0.1:5000](http://127.0.0.1:5000)
Future Improvements
Larger and more diverse labeled dataset

Cross-validation for more stable metrics

Switch to LinearSVC or LSTM for higher accuracy

Visualization of confidence scores

Database-backed storage instead of JSON file