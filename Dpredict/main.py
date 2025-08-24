# main.py
import os
import io
import json
from datetime import date
from typing import Optional, Dict, Any, List, Tuple

import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

# --- Matplotlib headless for server-side image generation ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- Gemini (Google) ---
try:
    from google import genai
    _USE_NEW_GENAI = True
except Exception:
    import google.generativeai as genai  # type: ignore
    _USE_NEW_GENAI = False

app = FastAPI(title="Nepal Drought Planner (CSV + Maps + Demand/Supply)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- In-memory state + files ----------------
LATEST: Dict[str, Any] = {
    "summary": None,
    "per_row": [],
    "last_uploaded_csv_cols": [],
    "districts": [],
    "df": None,  # cached DataFrame after upload
}

INVENTORY_FILE = "inventory.json"
DEMAND_FILE = "demand_history.json"  # stores user-input past-year demand MLD per district
STATIC_DIR = "static"
MAP_DIR = os.path.join(STATIC_DIR, "maps")
os.makedirs(MAP_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

def _load_json(path: str, default: Any) -> Any:
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(default, f, ensure_ascii=False, indent=2)
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _save_json(path: str, data: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_inventory():
    return _load_json(INVENTORY_FILE, {"items": {}})

def save_inventory(data):
    _save_json(INVENTORY_FILE, data)

def load_demand_history():
    # structure: {"records":[{"district":"Jhapa","year":2000,"demand_mld":120.0}]}
    return _load_json(DEMAND_FILE, {"records": []})

def save_demand_history(data):
    _save_json(DEMAND_FILE, data)

# ---------------- CSV schema ----------------
REQUIRED_COLS = [
    "Year",
    "Month",
    "District",
    "Rainfall_mm",
    "SoilMoisture",
    "Temperature_C",
    "Humidity_%",
    "NDVI",
    "Evapotranspiration_mm",
    "GroundwaterLevel_m",
    "Drought_Index",
    "Crop_Yield_Loss_%"
]

# Minimal lat/long for some Nepal districts (center-ish points).
DISTRICT_COORDS = {
    "jhapa": (26.64, 87.92),
    "morang": (26.62, 87.33),
    "sunsari": (26.65, 87.17),
    "ilam": (26.91, 87.93),
    "udayapur": (26.88, 86.63),
    "siraha": (26.67, 86.20),
    "saptari": (26.62, 86.70),
    "dhanusa": (26.83, 86.02),
    "mahottari": (26.65, 85.82),
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
    soil = _sf(row.get("SoilMoisture", 0))          # 0..1
    hum  = _sf(row.get("Humidity_%", 0)) / 100.0    # 0..1
    ndvi = _sf(row.get("NDVI", 0))                  # 0..1
    gw_m = _sf(row.get("GroundwaterLevel_m", 0))    # deeper worse

    water_def = max(0.0, (et - rain) / 150.0)
    hot = max(0.0, (temp - 30.0) / 15.0)
    low_soil = max(0.0, (0.5 - soil))
    low_hum = max(0.0, (0.6 - hum))
    low_ndvi = max(0.0, (0.5 - ndvi))
    deep_gw = min(1.0, gw_m / 20.0)

    idx = 0.35*water_def + 0.2*low_soil + 0.15*deep_gw + 0.15*hot + 0.1*low_ndvi + 0.05*low_hum
    return float(max(0.0, min(1.5, idx)))

def supply_proxy_raw(row: pd.Series) -> float:
    """
    Unitless supply proxy (higher is better):
    rainfall + shallow groundwater + humidity + greenness, scaled.
    """
    rain = _sf(row.get("Rainfall_mm", 0))
    gw_m = _sf(row.get("GroundwaterLevel_m", 0))
    hum  = _sf(row.get("Humidity_%", 0)) / 100.0
    ndvi = _sf(row.get("NDVI", 0))
    gw_avail = max(0.0, 1.0 - (gw_m / 20.0))  # ~0 when >20 m

    # scale rainfall and combine; result ~0..something
    return (rain / 200.0) * 0.45 + gw_avail * 0.30 + hum * 0.15 + ndvi * 0.10

def supply_demand_ratio(row: pd.Series) -> float:
    et = _sf(row.get("Evapotranspiration_mm", 0))
    temp = _sf(row.get("Temperature_C", 0))
    demand = (et / 200.0) * 0.65 + max(0.0, (temp - 20.0) / 20.0) * 0.35
    supply = supply_proxy_raw(row)
    return float(round(min(3.0, max(0.1, supply / max(1e-6, demand))), 3))

def irrigation_suggestion(row: pd.Series, risk: str) -> Dict[str, Any]:
    et = _sf(row.get("Evapotranspiration_mm", 0))
    soil = _sf(row.get("SoilMoisture", 0))  # 0..1 expected
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

# --------- Demand calibration (MLD) ----------
def _find_row_for_year(df: pd.DataFrame, district: str, year: int) -> Optional[pd.Series]:
    ddf = df[(df["District"].astype(str).str.strip().str.lower() == district.strip().lower()) &
             (df["Year"].astype("Int64") == year)]
    if ddf.empty:
        return None
    # Pick the median month row to be robust
    ddf = ddf.sort_values("Month")
    return ddf.iloc[len(ddf)//2]

def _calibrate_supply_to_mld(df: pd.DataFrame, district: str, year: int, demand_mld: float) -> Optional[float]:
    """
    Returns calibration factor K where  K * supply_proxy_raw(past_row) ~= demand_mld
    Then current_supply_mld = K * supply_proxy_raw(current_row)
    """
    past = _find_row_for_year(df, district, year)
    if past is None:
        return None
    proxy = supply_proxy_raw(past)
    if proxy <= 1e-9:
        return None
    return demand_mld / proxy

def _latest_row_for_district(df: pd.DataFrame, district: str) -> Optional[pd.Series]:
    ddf = df[df["District"].astype(str).str.strip().str.lower() == district.strip().lower()]
    if ddf.empty:
        return None
    ddf = ddf.sort_values(["Year", "Month"])
    return ddf.iloc[-1]

# ---------------- Routes ----------------
@app.get("/")
def root():
    return {"status": "ok", "message": "Nepal Drought Planner API running."}

@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Upload a CSV file.")

    raw = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {e}")

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing required columns: {missing}. Required: {REQUIRED_COLS}")

    # Normalize
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    df["Month"] = pd.to_numeric(df["Month"], errors="coerce").astype("Int64")
    df["District"] = df["District"].astype(str).str.strip()
    df["date"] = [_to_str_date(y, m) for y, m in zip(df["Year"], df["Month"])]

    # Metrics
    per_row: List[Dict[str, Any]] = []
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
            "humidity_pct": _sf(row.get("Humidity_%" , 0)),
            "ndvi": _sf(row.get("NDVI", 0)),
            "evapotranspiration_mm": _sf(row.get("Evapotranspiration_mm", 0)),
            "groundwater_m": _sf(row.get("GroundwaterLevel_m", 0)),
            "crop_yield_loss_pct": _sf(row.get("Crop_Yield_Loss_%", 0)),
            "irrigation": irr,
        })

    per_row.sort(key=lambda r: (r["date"], r["district"]))
    latest_date = per_row[-1]["date"]
    latest_rows = [r for r in per_row if r["date"] == latest_date]
    risk_order = {"Low": 0, "Medium": 1, "High": 2}
    worst_risk = max(latest_rows, key=lambda r: risk_order.get(r["risk"], 1))["risk"]
    avg_ratio = round(sum(r["supply_demand_ratio"] for r in per_row) / len(per_row), 3)

    alert = "✅ Conditions stable."
    if worst_risk == "High" or any(r["supply_demand_ratio"] < 0.9 for r in latest_rows):
        alert = "⚠️ Water shortage likely. Activate drought plan & rationing."
    elif worst_risk == "Medium":
        alert = "⚠️ Monitor conditions; prepare contingency irrigation."

    summary = {
        "current_date": latest_date,
        "current_risk": worst_risk,
        "avg_ratio": avg_ratio,
        "alert": alert,
    }

    LATEST["summary"] = summary
    LATEST["per_row"] = per_row
    LATEST["last_uploaded_csv_cols"] = list(df.columns)
    LATEST["districts"] = sorted(df["District"].dropna().astype(str).unique().tolist())
    LATEST["df"] = df

    return {"summary": summary, "per_row": per_row, "districts": LATEST["districts"]}

@app.get("/predict")
def predict_latest():
    if not LATEST["summary"]:
        raise HTTPException(status_code=404, detail="No CSV processed yet.")
    return LATEST

@app.get("/districts")
def districts():
    return {"districts": LATEST["districts"] or []}

# ---------------- Gemini AI recommendation ----------------
from pydantic import BaseModel
class AIRequest(BaseModel):
    climate_summary: str
    risk_level: str
    region: Optional[str] = None

@app.post("/ai/recommend")
def ai_recommend(req: AIRequest):
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "YOUR_API_KEY"
    if not api_key or api_key == "YOUR_API_KEY":
        raise HTTPException(status_code=400, detail="Set GEMINI_API_KEY environment variable.")

    system = (
        "You are an agronomy assistant for Nepal. Recommend crop choices "
        "and practices by LOW/MEDIUM/HIGH risk. Include short guidance on "
        "irrigation, drought-tolerant varieties, mulching, and planting windows. "
        "Keep under 180 words."
    )
    user = f"Region: {req.region or 'N/A'}\nRisk: {req.risk_level}\nClimate: {req.climate_summary}"

    try:
        if _USE_NEW_GENAI:
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[{"role": "user", "parts": [{"text": system + "\n\n" + user}]}]
            )
            text = response.text
        else:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-2.5-flash")
            response = model.generate_content(system + "\n\n" + user)
            text = response.text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini error: {e}")

    return {"recommendations": text}

# ---------------- Inventory ----------------
class AddStock(BaseModel):
    crop: str
    quantity_kg: float

class SellStock(BaseModel):
    crop: str
    quantity_kg: float

@app.get("/inventory")
def get_inventory():
    return load_inventory()

@app.post("/inventory/add")
def add_stock(item: AddStock):
    db = load_inventory()
    crop = item.crop.strip().lower()
    rec = db["items"].get(crop, {"stock_kg": 0.0, "sold_kg": 0.0})
    rec["stock_kg"] = round(rec["stock_kg"] + float(item.quantity_kg), 2)
    db["items"][crop] = rec
    save_inventory(db)
    return {"ok": True, "item": {crop: rec}}

@app.post("/inventory/sell")
def sell_stock(sale: SellStock):
    db = load_inventory()
    crop = sale.crop.strip().lower()
    if crop not in db["items"]:
        raise HTTPException(status_code=404, detail="Crop not found.")
    rec = db["items"][crop]
    if sale.quantity_kg > rec["stock_kg"]:
        raise HTTPException(status_code=400, detail="Not enough stock.")
    rec["stock_kg"] = round(rec["stock_kg"] - float(sale.quantity_kg), 2)
    rec["sold_kg"] = round(rec["sold_kg"] + float(sale.quantity_kg), 2)
    db["items"][crop] = rec
    save_inventory(db)
    return {"ok": True, "item": {crop: rec}}

# ---------------- Demand history & analysis ----------------
from pydantic import BaseModel

class DemandRecord(BaseModel):
    district: str
    year: int
    demand_mld: float  # Million Litres per Day

@app.post("/demand-history/add")
def add_demand(rec: DemandRecord):
    if LATEST["df"] is None:
        raise HTTPException(status_code=400, detail="Upload CSV first to align districts/years.")
    if rec.district.strip() not in LATEST["districts"]:
        raise HTTPException(status_code=400, detail="District not found in uploaded CSV.")

    db = load_demand_history()
    # upsert
    found = False
    for r in db["records"]:
        if r["district"].strip().lower() == rec.district.strip().lower() and int(r["year"]) == int(rec.year):
            r["demand_mld"] = float(rec.demand_mld)
            found = True
            break
    if not found:
        db["records"].append({"district": rec.district, "year": int(rec.year), "demand_mld": float(rec.demand_mld)})
    save_demand_history(db)
    return {"ok": True, "record": rec.dict()}

@app.get("/demand-history")
def list_demand(district: Optional[str] = None):
    db = load_demand_history()
    if district:
        filt = [r for r in db["records"] if r["district"].strip().lower() == district.strip().lower()]
        return {"records": filt}
    return db

@app.get("/analysis/demand-supply")
def demand_supply(district: str, use_latest_year: bool = True):
    """
    Uses user-entered past-year demand to calibrate supply MLD and estimates
    current surplus/shortage for the latest month available for the district.
    """
    if LATEST["df"] is None:
        raise HTTPException(status_code=404, detail="Upload CSV first.")
    df: pd.DataFrame = LATEST["df"]

    latest_row = _latest_row_for_district(df, district)
    if latest_row is None:
        raise HTTPException(status_code=404, detail="District not found in CSV.")

    # pick past record: prefer (latest_row.year - 1) else the closest available year from demand history
    dh = load_demand_history()
    recs = [r for r in dh["records"] if r["district"].strip().lower() == district.strip().lower()]
    if not recs:
        raise HTTPException(status_code=404, detail="No demand history for this district. Add one first.")

    target_year = int(latest_row["Year"]) - 1 if use_latest_year else recs[0]["year"]
    # choose exact-year record if exists, else nearest by absolute diff
    chosen = min(recs, key=lambda r: abs(int(r["year"]) - target_year))

    K = _calibrate_supply_to_mld(df, district, int(chosen["year"]), float(chosen["demand_mld"]))
    if K is None:
        raise HTTPException(status_code=400, detail="Could not calibrate from past-year data (missing or zero proxy).")

    current_proxy = supply_proxy_raw(latest_row)
    current_supply_mld = K * current_proxy
    # For demand, simplest is to reuse last year's MLD (or you could add growth % later)
    current_demand_mld = float(chosen["demand_mld"])

    surplus_mld = current_supply_mld - current_demand_mld
    status = "Surplus" if surplus_mld >= 0 else "Shortage"

    # also compute drought risk from that latest row
    raw_idx = latest_row.get("Drought_Index")
    idx = None if pd.isna(raw_idx) else float(raw_idx)
    if idx is None:
        idx = heuristic_drought_index(latest_row)
    risk = drought_risk_from_index(idx)

    return {
        "district": district,
        "latest_date": _to_str_date(int(latest_row["Year"]), int(latest_row["Month"])),
        "reference_year": int(chosen["year"]),
        "reference_demand_mld": round(float(chosen["demand_mld"]), 2),
        "estimated_current_supply_mld": round(float(current_supply_mld), 2),
        "assumed_current_demand_mld": round(float(current_demand_mld), 2),
        "surplus_mld": round(float(surplus_mld), 2),
        "status": status,
        "drought_risk": risk,
    }

# ---------------- Map generation ----------------
def _np_bounds():
    return (26.2, 30.5, 80.0, 88.5)  # lat_min, lat_max, lon_min, lon_max

def _risk_color(risk: str) -> str:
    return {"Low": "#7fd8be", "Medium": "#ffd166", "High": "#ef476f"}.get(risk, "#a8b2d1")

@app.get("/map")
def map_png(
    district: Optional[str] = Query(default=None),
    date_str: Optional[str] = Query(default=None),
):
    if LATEST["df"] is None:
        raise HTTPException(status_code=404, detail="Upload CSV first.")
    df: pd.DataFrame = LATEST["df"]

    if date_str:
        ddf = df[df["date"] == date_str].copy()
        if ddf.empty:
            return JSONResponse(status_code=404, content={"detail": "No rows for that date."})
    else:
        latest_date = df["date"].dropna().max()
        ddf = df[df["date"] == latest_date].copy()

    rows: List[Dict[str, Any]] = []
    for _, row in ddf.iterrows():
        raw_idx = row.get("Drought_Index")
        idx = None if pd.isna(raw_idx) else float(raw_idx)
        if idx is None:
            idx = heuristic_drought_index(row)
        risk = drought_risk_from_index(idx)
        name = str(row["District"]).strip()
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

    return FileResponse(out_path, media_type="image/png")

