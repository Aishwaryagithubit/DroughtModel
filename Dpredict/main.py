import os
import io
import json
from datetime import date
from typing import Optional, Dict, Any, List, Tuple

import pandas as pd
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import google.generativeai as genai

os.environ["GEMINI_API_KEY"] = "AIzaSyDN5pKtY57A9py8kOmBaK5CLWPW-02uG50"  # Replace with your Gemini API key

app = Flask(__name__)
CORS(app)

# ---------------- In-memory state + files ----------------
LATEST: Dict[str, Any] = {
    "summary": None,
    "per_row": [],
    "last_uploaded_csv_cols": [],
    "districts": [],
}

INVENTORY_FILE = "inventory.json"
DEMAND_FILE = "demand_history.json"
STATIC_DIR = "static"
MAP_DIR = os.path.join(STATIC_DIR, "maps")
os.makedirs(MAP_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

def _load_json(path: str, default: Any) -> Any:
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(default, f, ensure_ascii=False, indent=2)
        return default
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                raise ValueError("Empty file")
            return json.loads(content)
    except (json.JSONDecodeError, ValueError):
        print(f"Warning: {path} is corrupted. Recreating with default data.")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(default, f, ensure_ascii=False, indent=2)
        return default

def _save_json(path: str, data: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_inventory():
    return _load_json(INVENTORY_FILE, {"items": {}})

def save_inventory(data):
    _save_json(INVENTORY_FILE, data)

def load_demand_history():
    return _load_json(DEMAND_FILE, {"records": []})

def save_demand_history(data):
    _save_json(DEMAND_FILE, data)

# ---------------- CSV schema ----------------
REQUIRED_COLS = [
    "Year", "Month", "District", "Rainfall_mm", "SoilMoisture", "Temperature_C",
    "Humidity_%", "NDVI", "Evapotranspiration_mm", "GroundwaterLevel_m",
    "Drought_Index", "Crop_Yield_Loss_%"
]

DISTRICT_COORDS = {
    "jhapa": (26.64, 87.92), "morang": (26.62, 87.33), "sunsari": (26.65, 87.17),
    "ilam": (26.91, 87.93), "udayapur": (26.88, 86.63), "siraha": (26.67, 86.20),
    "saptari": (26.62, 86.70), "dhanusa": (26.83, 86.02), "mahottari": (26.65, 85.82),
    "sarlahi": (27.00, 85.57),
}

# ---------------- Helpers ----------------
def _sf(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return default

def drought_risk_from_index(idx: Optional[float]) -> str:
    if idx is None:
        return "Medium"
    if idx < 0.35:
        return "Low"
    if idx < 0.7:
        return "Medium"
    return "High"

def heuristic_drought_index(row: pd.Series) -> float:
    rain = _sf(row.get("Rainfall_mm", 0))
    et = _sf(row.get("Evapotranspiration_mm", 0))
    temp = _sf(row.get("Temperature_C", 0))
    soil = _sf(row.get("SoilMoisture", 0))
    hum = _sf(row.get("Humidity_%", 0)) / 100.0
    ndvi = _sf(row.get("NDVI", 0))
    gw_m = _sf(row.get("GroundwaterLevel_m", 0))

    water_def = max(0.0, (et - rain) / 150.0)
    hot = max(0.0, (temp - 30.0) / 15.0)
    low_soil = max(0.0, (0.5 - soil))
    low_hum = max(0.0, (0.6 - hum))
    low_ndvi = max(0.0, (0.5 - ndvi))
    deep_gw = min(1.0, gw_m / 20.0)

    idx = 0.35*water_def + 0.2*low_soil + 0.15*deep_gw + 0.15*hot + 0.1*low_ndvi + 0.05*low_hum
    return float(max(0.0, min(1.5, idx)))

def supply_proxy_raw(row: pd.Series) -> float:
    rain = _sf(row.get("Rainfall_mm", 0))
    gw_m = _sf(row.get("GroundwaterLevel_m", 0))
    hum = _sf(row.get("Humidity_%", 0)) / 100.0
    ndvi = _sf(row.get("NDVI", 0))
    gw_avail = max(0.0, 1.0 - (gw_m / 20.0))
    return (rain / 200.0) * 0.45 + gw_avail * 0.30 + hum * 0.15 + ndvi * 0.10

def supply_demand_ratio(row: pd.Series) -> float:
    et = _sf(row.get("Evapotranspiration_mm", 0))
    temp = _sf(row.get("Temperature_C", 0))
    demand = (et / 200.0) * 0.65 + max(0.0, (temp - 20.0) / 20.0) * 0.35
    supply = supply_proxy_raw(row)
    return float(round(min(3.0, max(0.1, supply / max(1e-6, demand))), 3))

def irrigation_suggestion(row: pd.Series, risk: str) -> Dict[str, Any]:
    et = _sf(row.get("Evapotranspiration_mm", 0))
    soil = _sf(row.get("SoilMoisture", 0))
    soil_pct = soil*100 if soil <= 1.5 else soil

    if risk == "High":
        times = 4
        depth = max(10, round(et * 1.2))
        note = "Prioritize high-value crops; mulch & drip if possible."
    elif risk == "Medium":
        times = 3
        depth = max(8, round(et))
        note = "Irrigate early/late; monitor soil moisture."
    else:
        times = 2
        depth = max(6, round(et * 0.8))
        note = "Maintain; consider deficit irrigation."

    if soil_pct > 60:
        note += " Soil OK; you may skip one event if rain is forecast."

    return {"events_per_week": times, "mm_per_event": int(depth), "notes": note}

def _to_str_date(y, m) -> str:
    try:
        y = int(y); m = int(m)
        return date(y, m, 1).isoformat()
    except Exception:
        return f"{y}-{int(m):02d}-01"

def _find_row_for_year(df: pd.DataFrame, district: str, year: int) -> Optional[pd.Series]:
    ddf = df[(df["district"].astype(str).str.strip().str.lower() == district.strip().lower()) &
             (df["year"].astype("Int64") == year)]
    if ddf.empty:
        return None
    ddf = ddf.sort_values("month")
    return ddf.iloc[len(ddf)//2]

def _latest_row_for_district(df: pd.DataFrame, district: str) -> Optional[pd.Series]:
    ddf = df[df["district"].astype(str).str.strip().str.lower() == district.strip().lower()]
    if ddf.empty:
        return None
    ddf = ddf.sort_values(["year", "month"])
    return ddf.iloc[-1]

def _calibrate_supply_to_mld(df: pd.DataFrame, district: str, year: int, demand_mld: float) -> Optional[float]:
    past = _find_row_for_year(df, district, year)
    if past is None:
        return None
    proxy = supply_proxy_raw(past)
    if proxy <= 1e-9:
        return None
    return demand_mld / proxy

# ---------------- Routes ----------------
@app.route("/", methods=["GET"])
def root():
    return jsonify({"status": "ok", "message": "Nepal Drought Planner API running."})

@app.route("/upload-csv", methods=["POST"])
def upload_csv():
    if 'file' not in request.files:
        return jsonify({"detail": "No file uploaded"}), 400
    
    file = request.files['file']
    if not file.filename.lower().endswith('.csv'):
        return jsonify({"detail": "Upload a CSV file"}), 400

    try:
        df = pd.read_csv(file)
    except Exception as e:
        return jsonify({"detail": f"Failed to read CSV: {e}"}), 400

    # Normalize column names to lowercase for validation
    df.columns = [col.strip().lower() for col in df.columns]
    required_cols_lower = [col.lower() for col in REQUIRED_COLS]
    missing = [col for col in required_cols_lower if col not in df.columns]
    if missing:
        return jsonify({"detail": f"Missing required columns: {missing}. Required: {REQUIRED_COLS}"}), 400

    # Rename columns to match REQUIRED_COLS
    col_map = {col.lower(): col for col in REQUIRED_COLS}
    df = df.rename(columns=col_map)

    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    df["Month"] = pd.to_numeric(df["Month"], errors="coerce").astype("Int64")
    df["District"] = df["District"].astype(str).str.strip()
    df["date"] = [_to_str_date(y, m) for y, m in zip(df["Year"], df["Month"])]

    per_row = []
    for _, row in df.iterrows():
        raw_idx = row.get("Drought_Index")
        idx = None if pd.isna(raw_idx) else float(raw_idx)
        if idx is None:
            idx = heuristic_drought_index(row)

        risk = drought_risk_from_index(idx)
        ratio = supply_demand_ratio(row)
        wb = _sf(row.get("Rainfall_mm", 0)) - _sf(row.get("Evapotranspiration_mm", 0))
        irr = irrigation_suggestion(row, risk)

        per_row.append({
            "date": row["date"],
            "year": int(row["Year"]) if pd.notna(row["Year"]) else None,
            "month": int(row["Month"]) if pd.notna(row["Month"]) else None,
            "district": row["District"],
            "drought_index": round(idx, 3),
            "risk": risk,
            "water_balance_mm": round(wb, 2),
            "supply_demand_ratio": ratio,
            "rainfall_mm": _sf(row.get("Rainfall_mm", 0)),
            "temperature_c": _sf(row.get("Temperature_C", 0)),
            "soil_moisture": _sf(row.get("SoilMoisture", 0)),
            "humidity_pct": _sf(row.get("Humidity_%", 0)),
            "ndvi": _sf(row.get("NDVI", 0)),
            "evapotranspiration_mm": _sf(row.get("Evapotranspiration_mm", 0)),
            "groundwater_m": _sf(row.get("GroundwaterLevel_m", 0)),
            "crop_yield_loss_pct": _sf(row.get("Crop_Yield_Loss_%", 0)),
            "irrigation": irr,
        })

    per_row.sort(key=lambda r: (r["date"], r["district"]))
    latest_date = per_row[-1]["date"] if per_row else None
    latest_rows = [r for r in per_row if r["date"] == latest_date] if per_row else []
    risk_order = {"Low": 0, "Medium": 1, "High": 2}
    worst_risk = max(latest_rows, key=lambda r: risk_order.get(r["risk"], 1))["risk"] if latest_rows else "-"
    avg_ratio = round(sum(r["supply_demand_ratio"] for r in per_row) / len(per_row), 3) if per_row else "-"

    alert = "✅ Conditions stable." if per_row else "No data yet."
    if per_row and (worst_risk == "High" or any(r["supply_demand_ratio"] < 0.9 for r in latest_rows)):
        alert = "⚠️ Water shortage likely. Activate drought plan & rationing."
    elif per_row and worst_risk == "Medium":
        alert = "⚠️ Monitor conditions; prepare contingency irrigation."

    summary = {
        "current_date": latest_date if per_row else "-",
        "current_risk": worst_risk,
        "avg_ratio": avg_ratio,
        "alert": alert,
    }

    LATEST["summary"] = summary
    LATEST["per_row"] = per_row
    LATEST["last_uploaded_csv_cols"] = list(df.columns) if per_row else []
    LATEST["districts"] = sorted(df["District"].dropna().astype(str).unique().tolist()) if per_row else []

    return jsonify({"summary": summary, "per_row": per_row, "districts": LATEST["districts"]})

@app.route("/predict", methods=["GET"])
def predict_latest():
    if not LATEST["summary"]:
        return jsonify({
            "summary": {
                "current_date": "-",
                "current_risk": "-",
                "avg_ratio": "-",
                "alert": "No data yet."
            },
            "per_row": [],
            "districts": [],
            "last_uploaded_csv_cols": []
        })
    
    return jsonify({
        "summary": LATEST["summary"],
        "per_row": LATEST["per_row"],
        "districts": LATEST["districts"],
        "last_uploaded_csv_cols": LATEST["last_uploaded_csv_cols"]
    })

@app.route("/districts", methods=["GET"])
def districts():
    return jsonify({"districts": LATEST["districts"] or []})

@app.route("/ai/tips", methods=["POST"])
def ai_tips():
    data = request.get_json()
    climate_summary = data.get("climate_summary", "")
    region = data.get("region")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return jsonify({"detail": "Set GEMINI_API_KEY environment variable."}), 400

    system = (
        "You are a water management advisor for Nepal. Provide short, actionable, "
        "localized water-saving tips for farmers and communities. Keep it under 150 words, "
        "in clear bullet-style sentences."
    )
    user = f"Region: {region or 'N/A'}\nClimate: {climate_summary}"

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(system + "\n\n" + user)
        text = response.text
    except Exception as e:
        return jsonify({"detail": f"Gemini error: {e}"}), 500

    return jsonify({"tips": text})

@app.route("/ai/crop-recommendations", methods=["POST"])
def ai_crop_recommendations():
    data = request.get_json()
    region = data.get("region", "")
    climate_data = data.get("climate_data", {})
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return jsonify({"detail": "GEMINI_API_KEY environment variable not set."}), 400
    
    prompt = f"""
    As an agricultural expert for Nepal, provide crop recommendations based on these conditions:
    
    Region: {region or 'Not specified'}
    Climate data: {json.dumps(climate_data, indent=2)}
    
    Please provide:
    1. 3-5 suitable crops for this region and conditions
    2. Brief explanation why each crop is suitable
    3. Planting season recommendations
    4. Water requirements and drought tolerance
    5. Any special considerations for Nepali context
    
    Format the response in clear, concise bullet points.
    """
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        text = response.text
    except Exception as e:
        return jsonify({"detail": f"Gemini error: {e}"}), 500

    return jsonify({"recommendations": text})

@app.route("/inventory", methods=["GET"])
def get_inventory():
    return jsonify(load_inventory())

@app.route("/inventory/add", methods=["POST"])
def add_stock():
    data = request.get_json()
    crop = data.get("crop", "").strip().lower()
    quantity_kg = float(data.get("quantity_kg", 0))
    
    if quantity_kg <= 0:
        return jsonify({"detail": "Quantity must be positive."}), 400

    db = load_inventory()
    rec = db["items"].get(crop, {"stock_kg": 0.0, "sold_kg": 0.0})
    rec["stock_kg"] = round(rec["stock_kg"] + quantity_kg, 2)
    db["items"][crop] = rec
    save_inventory(db)
    return jsonify({"ok": True, "item": {crop: rec}})

@app.route("/inventory/sell", methods=["POST"])
def sell_stock():
    data = request.get_json()
    crop = data.get("crop", "").strip().lower()
    quantity_kg = float(data.get("quantity_kg", 0))
    
    if quantity_kg <= 0:
        return jsonify({"detail": "Quantity must be positive."}), 400

    db = load_inventory()
    if crop not in db["items"]:
        return jsonify({"detail": "Crop not found."}), 404
        
    rec = db["items"][crop]
    if quantity_kg > rec["stock_kg"]:
        return jsonify({"detail": "Not enough stock."}), 400
        
    rec["stock_kg"] = round(rec["stock_kg"] - quantity_kg, 2)
    rec["sold_kg"] = round(rec["sold_kg"] + quantity_kg, 2)
    db["items"][crop] = rec
    save_inventory(db)
    return jsonify({"ok": True, "item": {crop: rec}})

@app.route("/demand-history/add", methods=["POST"])
def add_demand():
    data = request.get_json()
    district = data.get("district", "").strip()
    year = int(data.get("year", 0))
    demand_mld = float(data.get("demand_mld", 0))
    
    if not LATEST["per_row"]:
        return jsonify({"detail": "Upload CSV first to align districts/years."}), 400
        
    if district not in LATEST["districts"]:
        return jsonify({"detail": "District not found in uploaded CSV."}), 400

    db = load_demand_history()
    found = False
    for r in db["records"]:
        if r["district"].strip().lower() == district.lower() and int(r["year"]) == year:
            r["demand_mld"] = demand_mld
            found = True
            break
    if not found:
        db["records"].append({"district": district, "year": year, "demand_mld": demand_mld})
    save_demand_history(db)
    return jsonify({"ok": True, "record": {"district": district, "year": year, "demand_mld": demand_mld}})

@app.route("/demand-history", methods=["GET"])
def list_demand():
    district = request.args.get("district")
    db = load_demand_history()
    if district:
        filt = [r for r in db["records"] if r["district"].strip().lower() == district.strip().lower()]
        return jsonify({"records": filt})
    return jsonify(db)

@app.route("/analysis/demand-supply", methods=["GET"])
def demand_supply():
    district = request.args.get("district")
    use_latest_year = request.args.get("use_latest_year", "true").lower() == "true"
    
    if not district:
        return jsonify({"detail": "District parameter required"}), 400
        
    if not LATEST["per_row"]:
        return jsonify({"detail": "Upload CSV first."}), 404
        
    df = pd.DataFrame(LATEST["per_row"])
    
    latest_row = _latest_row_for_district(df, district)
    if latest_row is None:
        return jsonify({"detail": "District not found in CSV."}), 404

    dh = load_demand_history()
    recs = [r for r in dh["records"] if r["district"].strip().lower() == district.strip().lower()]
    if not recs:
        return jsonify({"detail": "No demand history for this district. Add one first."}), 404

    target_year = int(latest_row["year"]) - 1 if use_latest_year else recs[0]["year"]
    chosen = min(recs, key=lambda r: abs(int(r["year"]) - target_year))

    K = _calibrate_supply_to_mld(df, district, int(chosen["year"]), float(chosen["demand_mld"]))
    if K is None:
        return jsonify({"detail": "Could not calibrate from past-year data (missing or zero proxy)."}), 400

    current_proxy = supply_proxy_raw(latest_row)
    current_supply_mld = K * current_proxy
    current_demand_mld = float(chosen["demand_mld"])
    surplus_mld = current_supply_mld - current_demand_mld
    status = "Surplus" if surplus_mld >= 0 else "Shortage"

    raw_idx = latest_row.get("drought_index")
    idx = None if pd.isna(raw_idx) else float(raw_idx)
    if idx is None:
        idx = heuristic_drought_index(latest_row)
    risk = drought_risk_from_index(idx)

    return jsonify({
        "district": district,
        "latest_date": _to_str_date(int(latest_row["year"]), int(latest_row["month"])),
        "reference_year": int(chosen["year"]),
        "reference_demand_mld": round(float(chosen["demand_mld"]), 2),
        "estimated_current_supply_mld": round(float(current_supply_mld), 2),
        "assumed_current_demand_mld": round(float(current_demand_mld), 2),
        "surplus_mld": round(float(surplus_mld), 2),
        "status": status,
        "drought_risk": risk,
    })

def _np_bounds():
    return (26.2, 30.5, 80.0, 88.5)

def _risk_color(risk: str) -> str:
    return {"Low": "#7fd8be", "Medium": "#ffd166", "High": "#ef476f"}.get(risk, "#a8b2d1")

@app.route("/map", methods=["GET"])
def map_png():
    district = request.args.get("district")
    date_str = request.args.get("date_str")
    
    if not LATEST["per_row"]:
        return jsonify({"detail": "Upload CSV first."}), 404
        
    df = pd.DataFrame(LATEST["per_row"])
    
    if date_str:
        ddf = df[df["date"] == date_str].copy()
        if ddf.empty:
            return jsonify({"detail": "No rows for that date."}), 404
    else:
        latest_date = df["date"].dropna().max() if not df["date"].dropna().empty else None
        if not latest_date:
            return jsonify({"detail": "No valid date in data."}), 404
        ddf = df[df["date"] == latest_date].copy()

    rows = []
    for _, row in ddf.iterrows():
        raw_idx = row.get("drought_index")
        idx = None if pd.isna(raw_idx) else float(raw_idx)
        if idx is None:
            idx = heuristic_drought_index(row)
        risk = drought_risk_from_index(idx)
        name = str(row["district"]).strip()
        rows.append({"district": name, "idx": idx, "risk": risk})

    lat_min, lat_max, lon_min, lon_max = _np_bounds()
    fig = plt.figure(figsize=(6, 5), dpi=150)
    ax = plt.gca()
    title_date = ddf.iloc[0]["date"] if not ddf.empty else date.today().isoformat()
    ax.set_title(f"Nepal — Drought Risk Map ({title_date})", fontsize=9)
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, alpha=0.2)

    seen_any = False
    for r in rows:
        key = r["district"].strip().lower()
        if key in DISTRICT_COORDS:
            lat, lon = DISTRICT_COORDS[key]
            ax.scatter(lon, lat, s=35, color=_risk_color(r["risk"]), edgecolor="#203047", linewidth=0.5, alpha=0.95)
            seen_any = True

    if district:
        k = district.strip().lower()
        if k in DISTRICT_COORDS:
            lat, lon = DISTRICT_COORDS[k]
            ax.scatter(lon, lat, s=120, facecolors="none", edgecolors="#ffffff", linewidth=2.0)
            ax.text(lon + 0.1, lat + 0.1, district, fontsize=8, weight="bold", color="#ffffff",
                    bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.35))
        else:
            ax.text(lon_min + 0.2, lat_min + 0.3, f"Coordinates unknown for: {district}", color="#ffdddd", fontsize=8)

    if not seen_any:
        ax.text(lon_min + 0.2, lat_min + 0.3, "No known coordinates for districts in this CSV/date.",
                color="#ffdddd", fontsize=8)

    for i, lbl in enumerate(["Low", "Medium", "High"]):
        ax.scatter(lon_min + 0.6 + i*1.0, lat_max - 0.4, s=40, color=_risk_color(lbl), edgecolor="#203047", linewidth=0.5)
        ax.text(lon_min + 0.75 + i*1.0, lat_max - 0.45, lbl, fontsize=7, color="#dfe9ff")

    safe_d = (district or "all").replace("/", "_").replace("\\", "_").replace(" ", "_")
    safe_date = title_date.replace(":", "_")
    out_path = os.path.join(MAP_DIR, f"map_{safe_date}_{safe_d}.png")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    return send_file(out_path, mimetype='image/png')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)