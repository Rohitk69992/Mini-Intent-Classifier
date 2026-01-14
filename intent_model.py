import joblib
from preprocess import clean_text

vectorizer = joblib.load("vectorizer.pkl")
model = joblib.load("model.pkl")

def predict(text: str):
    clean = clean_text(text)
    X = vectorizer.transform([clean])
    probs = model.predict_proba(X)[0]
    idx = probs.argmax()
    return clean, model.classes_[idx], float(probs[idx])
