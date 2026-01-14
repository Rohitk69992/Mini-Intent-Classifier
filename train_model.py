import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from app.preprocess import clean_text

# ----------------------------
# Load & preprocess data
# ----------------------------
df = pd.read_csv("data/train.csv")
df["clean_text"] = df["text"].apply(clean_text)

X = df["clean_text"]
y = df["label"]

# ----------------------------
# Train / test split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# ----------------------------
# Vectorization (IMPORTANT TUNING)
# ----------------------------
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),      # capture intent phrases
    min_df=2,                # remove very rare noise terms
    max_df=0.85,             # remove overly common terms
    sublinear_tf=True        # dampen frequent words
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ----------------------------
# Model (BETTER REGULARIZATION)
# ----------------------------
model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    C=2.0,                   # slightly less regularization
    solver="liblinear"       # works well for small text data
)

model.fit(X_train_vec, y_train)

# ----------------------------
# Evaluation
# ----------------------------
y_pred = model.predict(X_test_vec)

accuracy = accuracy_score(y_test, y_pred)

print("\nMODEL EVALUATION")
print("----------------")
print(f"Accuracy: {accuracy:.2f}\n")
print(classification_report(y_test, y_pred))

# ----------------------------
# Save artifacts
# ----------------------------
joblib.dump(vectorizer, "vectorizer.pkl")
joblib.dump(model, "model.pkl")

print("Model and vectorizer saved.")
