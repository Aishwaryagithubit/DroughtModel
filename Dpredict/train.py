# train.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os

# Load dataset
df = pd.read_csv("Drought_Dataset.csv")

# Create binary target for drought
df["Drought"] = df["Drought_Index"].apply(lambda x: 1 if x > 0.5 else 0)

# Features and target
X = df.drop(["Drought", "Drought_Index", "Crop_Yield_Loss_%"], axis=1)
X = pd.get_dummies(X, drop_first=True)  # encode categorical
y = df["Drought"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train RandomForest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save trained model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/drought_clf.joblib")
print("✅ Model trained and saved successfully!")
