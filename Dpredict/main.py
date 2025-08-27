import streamlit as st
import pandas as pd
import joblib
import folium
from streamlit_folium import st_folium

# Load trained model and encoder
try:
    model = joblib.load("model.pkl")
    encoder = joblib.load("encoder.pkl")
except:
    st.error("❌ Model or encoder not found. Please run train.py first.")
    st.stop()

st.set_page_config(page_title="🌾 Drought Prediction Dashboard", layout="wide")
st.title("🌾 Drought Prediction Dashboard")

# Upload CSV
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("📊 Dataset Preview:", data.head())

    if "District" not in data.columns:
        st.error("❌ CSV must contain a 'District' column.")
    else:
        # Select District
        district = st.selectbox("Select District", data["District"].unique())
        district_encoded = encoder.transform([district])[0]

        # Latest values for prediction
        latest = data[data["District"] == district].iloc[-1]
        features = [[
            latest["Year"],
            latest["Month"],
            district_encoded,
            latest["Rainfall_mm"],
            latest["SoilMoisture"],
            latest["Temperature_C"]
        ]]

        # Predict drought risk (rainfall level used as proxy)
        prediction = model.predict(features)[0]
        st.metric(label="🌧️ Predicted Rainfall (mm)", value=f"{prediction:.2f}")

        # Crop recommendation
        if prediction < 50:
            st.warning("⚠️ Low rainfall predicted. Recommended crops: Millet, Sorghum")
        elif 50 <= prediction < 150:
            st.info("✅ Moderate rainfall predicted. Recommended crops: Maize, Wheat")
        else:
            st.success("🌿 High rainfall predicted. Recommended crops: Rice, Sugarcane")

        # Map visualization
        st.subheader("🗺️ District Map (Demo)")
        # For simplicity, randomize coordinates slightly for each district (optional)
        coords = {
            "Kathmandu": [27.7, 85.3],
            "Pokhara": [28.2, 83.9],
            "Biratnagar": [26.5, 87.3]
        }
        lat, lon = coords.get(district, [27.7, 85.3])
        m = folium.Map(location=[lat, lon], zoom_start=7)
        folium.Marker([lat, lon], popup=district).add_to(m)
        st_folium(m, width=700, height=500)
else:
    st.info("ℹ️ Please upload a CSV file to start predictions.")
