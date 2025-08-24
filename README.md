🌾 Drought Prediction & Water Management Web App

A Flask-based web application that leverages AI (Google Gemini) and climate data to predict drought risks, recommend crops, analyze water demand vs. supply, and promote sustainable agriculture practices.

This project was built as a portfolio piece to demonstrate expertise in data science, AI integration, and full-stack web development.

🚀 Features
🔍 Drought Risk Prediction

Upload climate dataset (CSV) containing rainfall, soil moisture, temperature, humidity, NDVI, groundwater level, etc.

Predicts drought risk levels (Low, Medium, High) using climate features.

AI-generated crop recommendations and irrigation scheduling via Google Gemini API.

Provides alerts when high drought risk is detected.

💧 Water Demand & Supply Analysis

Input past year’s water demand and supply.

System analyzes deficit/surplus conditions.

Visual chart comparison (matplotlib).

Awareness section with AI-generated water-saving tips.

🌍 Map Visualization

Interactive map (matplotlib-based) of Nepal districts.

Color-coded drought risks:

🟢 Low Risk

🟠 Medium Risk

🔴 High Risk

🌱 Crop Stock & Sales Tracker

Manage crop stock (e.g., Rice 100kg).

Record sales and update available inventory.

Dashboard-style display of crop availability.

📡 AI-Powered Insights

Integrated with Google Gemini AI for:

Irrigation scheduling.

Awareness tips (water conservation, sustainable farming).

Contextual guidance for farmers.

🛠️ Tech Stack

Backend: Flask (Python)

Frontend: HTML, Bootstrap (basic templates)

Data Processing: Pandas, NumPy

Visualization: Matplotlib

AI Integration: Google Gemini (google-generativeai SDK)

Other Tools: Pydantic (v1), Jinja2

📊 Usage Flow

Upload Dataset – Choose Drought_Dataset.csv.

Select District & Month – Example: Jhapa, Jan 2000.

View Predictions – Drought risk, crop suggestions, irrigation schedule.

Enter Water Demand/Supply – Get deficit/surplus insights.

Check Awareness Tips – AI-powered water-saving practices.

Track Crop Stock & Sales – Add stock, record sales, monitor inventory.

View Map Visualization – Risk heatmap across Nepal districts.
