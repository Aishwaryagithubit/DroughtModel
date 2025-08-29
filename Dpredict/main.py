from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import os
import google.generativeai as genai

# =========================
# Flask App Setup
# =========================
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

# Load trained ML model
MODEL_PATH = "models/drought_clf.joblib"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("❌ Model file not found! Run train.py first.")
model = joblib.load(MODEL_PATH)

print("Type of loaded object:", type(model))

# Configure Gemini API (replace value with your real key)
GEMINI_API_KEY = "AIzaSyDN5pKtY57A9py8kOmBaK5CLWPW-02uG50"
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.5-flash")

# Store uploaded data temporarily
uploaded_data = None

# =========================
# Routes
# =========================
@app.route("/upload", methods=["POST"])
def upload_file():
    global uploaded_data
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    # Read CSV into pandas DataFrame
    uploaded_data = pd.read_csv(file)

    # Preview first 5 rows
    preview = uploaded_data.head().to_dict(orient="records")

    # Collect districts if column exists
    districts = []
    if "District" in uploaded_data.columns:
        districts = sorted(uploaded_data["District"].dropna().unique().tolist())

    return jsonify({"message": "✅ File uploaded successfully", "preview": preview, "districts": districts})


@app.route("/predict", methods=["POST"])
def predict():
    global uploaded_data
    if uploaded_data is None:
        return jsonify({"error": "No data uploaded"}), 400

    data = request.json or {}
    district = data.get("district")
    if not district:
        return jsonify({"error": "No district provided"}), 400

    if "District" not in uploaded_data.columns:
        return jsonify({"error": "CSV has no 'District' column"}), 400

    row = uploaded_data[uploaded_data["District"] == district]
    if row.empty:
        return jsonify({"error": f"District '{district}' not found"}), 404

    # Drop target column if exists
    features = row.drop("Drought", axis=1, errors="ignore")

    # Apply same preprocessing as training
    features = pd.get_dummies(features, drop_first=True)

    # Align with model training columns
    train_cols = model.feature_names_in_  # sklearn attribute
    for col in train_cols:
        if col not in features:
            features[col] = 0  # add missing column with zeros
    features = features[train_cols]  # reorder columns exactly like training

    # Predict
    try:
        prediction_raw = model.predict(features)[0]
        label = "No Drought" if int(prediction_raw) == 0 else "Drought"
    except Exception as e:
        return jsonify({"error": "Model prediction failed", "details": str(e)}), 500

    # Optional probability
    prob_value = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features)[0]
        prob_value = float(proba[-1])  # probability of drought

    resp = {"district": district, "prediction": label}
    if prob_value is not None:
        resp["probability"] = round(prob_value, 3)

    return jsonify(resp)



@app.route("/ai-advice", methods=["POST"])
def ai_advice():
    data = request.json or {}
    user_input = data.get("query", "")

    if not user_input:
        return jsonify({"error": "No query provided"}), 400

    # Generate response from Gemini
    try:
        response = gemini_model.generate_content(
            f"Give detailed crop recommendation for the following situation:\n{user_input}"
        )
        advice_text = getattr(response, "text", None) or response.get("candidates", [{}])[0].get("content", "")
    except Exception as e:
        return jsonify({"error": "AI service failed", "details": str(e)}), 500

    return jsonify({"advice": advice_text})


@app.route("/recommendations", methods=["GET"])
def recommendations():
    tips = [
        "💧 Use drip irrigation to save water.",
        "🌱 Plant drought-resistant crops.",
        "📊 Monitor soil moisture regularly.",
        "🕰️ Irrigate crops in the early morning or late evening.",
        "🔄 Practice crop rotation to maintain soil health."
    ]
    return jsonify({"recommendations": tips})


# Run App
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)


