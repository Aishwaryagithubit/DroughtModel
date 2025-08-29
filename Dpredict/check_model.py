import joblib

MODEL_PATH = "models/drought_clf.joblib"

# Load the model
model = joblib.load(MODEL_PATH)

# Check the type
print("Type of loaded object:", type(model))

# Optional: check some attributes
if hasattr(model, "n_features_in_"):
    print("Number of features used in training:", model.n_features_in_)
else:
    print("❌ This is not a trained sklearn model!")
