import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Load dataset
data = pd.read_csv("Drought_Dataset.csv")

# Encode District column
le = LabelEncoder()
data["District_encoded"] = le.fit_transform(data["District"])

# Features (X) and Target (y)
X = data[["Year", "Month", "District_encoded", "Rainfall_mm", "SoilMoisture", "Temperature_C"]]
y = data["Rainfall_mm"]  # predicting rainfall as drought indicator (you can change target if needed)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train RandomForest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model + encoder
joblib.dump(model, "model.pkl")
joblib.dump(le, "encoder.pkl")

print("✅ Model and encoder saved successfully!")

