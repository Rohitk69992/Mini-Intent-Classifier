from flask import Flask, request, jsonify, render_template
from datetime import datetime
import json
import os

from intent_model import predict

app = Flask(__name__)

RESULTS_PATH = "results/predictions.json"
os.makedirs("results", exist_ok=True)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict_intent():
    data = request.get_json()
    text = data.get("text", "")

    clean_text, intent, confidence = predict(text)

    entry = {
        "text": text,
        "clean_text": clean_text,
        "intent": intent,
        "confidence": round(confidence, 2),
        "timestamp": datetime.utcnow().isoformat()
    }

    # Ensure file exists
    if not os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, "w") as f:
            json.dump([], f)

    # SAFE JSON LOAD
    try:
        with open(RESULTS_PATH, "r") as f:
            predictions = json.load(f)
    except json.JSONDecodeError:
        predictions = []

    predictions.append(entry)

    with open(RESULTS_PATH, "w") as f:
        json.dump(predictions, f, indent=2)

    return jsonify(entry)


if __name__ == "__main__":
    app.run(debug=True)
