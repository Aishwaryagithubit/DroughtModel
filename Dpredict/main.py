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

# Configure Gemini API (replace value with your real key)
GEMINI_API_KEY = "AIzaSyDN5pKtY57A9py8kOmBaK5CLWPW-02uG50"
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-pro")

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
    """
    Expects JSON body: { "district": "<district-name>" }
    Returns:
      {
        "district": "<district-name>",
        "prediction": "Drought" | "No Drought",
        "probability": 0.74   # optional, probability of class 1 (Drought) if model supports predict_proba
      }
    """
    global uploaded_data
    if uploaded_data is None:
        return jsonify({"error": "No data uploaded"}), 400

    data = request.json or {}
    district = data.get("district")
    if not district:
        return jsonify({"error": "No district provided"}), 400

    if "District" not in uploaded_data.columns:
        return jsonify({"error": "CSV has no 'District' column"}), 400

    # Filter the row(s) for the district
    row = uploaded_data[uploaded_data["District"] == district]
    if row.empty:
        return jsonify({"error": f"District '{district}' not found"}), 404

    # Drop target column if exists
    features = row.drop("Drought", axis=1, errors="ignore")

    # Model prediction
    try:
        prediction_raw = model.predict(features)[0]
    except Exception as e:
        return jsonify({"error": "Model prediction failed", "details": str(e)}), 500

    label = "No Drought" if int(prediction_raw) == 0 else "Drought"

    # Try to get probability for class '1' if available
    prob_value = None
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(features)[0]  # array like [p_class0, p_class1] or in different order
            # find index for class '1' if model.classes_ exists
            if hasattr(model, "classes_"):
                classes = list(model.classes_)
                if 1 in classes:
                    idx = classes.index(1)
                    prob_value = float(proba[idx])
                elif '1' in classes:
                    idx = classes.index('1')
                    prob_value = float(proba[idx])
                else:
                    # Fallback: if classes are [0,1] assume index 1 is drought
                    if len(proba) >= 2:
                        prob_value = float(proba[-1])
            else:
                # If no classes_ attribute, assume second column is 'Drought'
                if len(proba) >= 2:
                    prob_value = float(proba[-1])
    except Exception:
        # If probability computation fails, ignore and continue returning label only.
        prob_value = None

    resp = {"district": district, "prediction": label}
    if prob_value is not None:
        # clamp to 0..1 and return
        prob_value = max(0.0, min(1.0, float(prob_value)))
        resp["probability"] = prob_value

    return jsonify(resp)


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

# Run App
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)


