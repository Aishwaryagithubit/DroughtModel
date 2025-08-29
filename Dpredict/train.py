import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import os


# Load Dataset

DATA_PATH = "Drought_Dataset.csv"
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH} - please place your CSV there.")

df = pd.read_csv(DATA_PATH)

print("Columns in dataset:", df.columns.tolist())
print(df.head())

# Ensure target exists
if "Drought" not in df.columns:
    raise ValueError("Dataset must contain a 'Drought' column as target (0/1).")

# Features & target
X = df.drop("Drought", axis=1)
y = df["Drought"]

# Convert non-numeric columns if any (simple handling)
# NOTE: If you have categorical features, do proper preprocessing. This is minimal fallback:
X = pd.get_dummies(X, drop_first=True)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Train Model

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print("=== Drought Classifier Report ===")
print(classification_report(y_test, y_pred))


# Save Model
os.makedirs("models", exist_ok=True)
joblib.dump(clf, "models/drought_clf.joblib")
print("✅ Model saved at models/drought_clf.joblib")

