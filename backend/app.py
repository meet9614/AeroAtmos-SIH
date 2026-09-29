"""
app.py — AeroAtmos backend (AI-Powered Hyperlocal Weather Downscaling)
FastAPI backend for AeroAtmos: Panchayat-level weather intelligence.

Endpoints
---------
GET  /districts
GET  /blocks/{district}
GET  /panchayats/{block}
GET  /forecast/{panchayat_id}?month=7
GET  /forecast/block/{block}?month=7      (all panchayats in a block, for the map)
GET  /model/skill                          (validation vs. baselines, from train_model.py)

Run (from the repo root or from backend/):
    pip install -r backend/requirements.txt
    python backend/data_generator.py
    python backend/train_model.py
    uvicorn app:app --reload --app-dir backend
Then open http://127.0.0.1:8000/docs
"""

import json
import zlib
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

BASE = Path(__file__).resolve().parent

app = FastAPI(title="AeroAtmos API — AI-Powered Hyperlocal Weather Downscaling", version="0.2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])

master = pd.read_csv(BASE / "panchayat_master.csv")
bundle = joblib.load(BASE / "downscaling_model.joblib")
rain_model = bundle["rain_correction"]    # RandomForest on the block-forecast correction
rain_model.named_steps["reg"].n_jobs = 1  # small requests: threading overhead > gain
temp_model = bundle["temp_correction"]    # lapse-rate style correction on elevation

MONSOON = (6, 7, 8, 9)
RAIN_DAY_MM = 2.5  # IMD: "light rain" starts at 2.5 mm in 24 hours

# IMD terminology for 24-hour rainfall: (upper limit in mm, name)
IMD_RAIN_CATEGORIES = [
    (0.1, "No rain"), (2.5, "Very light rain"), (15.6, "Light rain"),
    (64.5, "Moderate rain"), (115.6, "Heavy rain"), (204.5, "Very heavy rain"),
    (float("inf"), "Extremely heavy rain"),
]


def rain_category(mm: float) -> str:
    for upper, name in IMD_RAIN_CATEGORIES:
        if mm < upper:
            return name


def fetch_block_forecast(block: str, month: int) -> dict:
    """
    Stand-in for the real IMD block-level API call. Replace this with a
    request to IMD's block forecast feed.

    Seeded with a stable checksum of block + month, so the demo shows the same
    numbers after every restart (Python's hash() changes per process), and drawn
    from the same distribution as the training data in data_generator.py.
    """
    rng = np.random.default_rng(zlib.crc32(f"{block}|{month}".encode()))
    season_factor = 1.6 if month in MONSOON else 0.5
    rain = max(rng.normal(18 * season_factor, 6), 0)
    temp = rng.normal(30 if month not in (12, 1) else 20, 3)
    return {"block_forecast_rain_mm": round(float(rain), 1),
            "block_forecast_temp_c": round(float(temp), 1)}


def advisory_for(rain: float, temp: float, humidity: float) -> list[str]:
    category = rain_category(rain)
    tips = []
    if rain >= 64.5:
        tips += [f"{category} expected — do not irrigate",
                 "Postpone pesticide/fertilizer application",
                 "Clear field drainage channels and protect harvested produce"]
    elif rain >= 15.6:
        tips += [f"{category} expected — skip irrigation",
                 "Hold off on spraying; chemicals would wash off"]
    elif rain >= RAIN_DAY_MM:
        tips += [f"{category} expected — reduce irrigation",
                 "Spray only during dry spells"]
    else:
        tips.append(f"{category} expected — irrigate as per crop need")

    if temp >= 36 and rain < RAIN_DAY_MM:
        tips += ["Hot, dry conditions — irrigate in the early morning or evening",
                 "Avoid spraying during peak afternoon heat"]
    if humidity >= 85:
        tips.append("High humidity — watch crops for fungal disease")
    return tips


def predict_rows(rows: pd.DataFrame, month: int) -> list[dict]:
    """Downscaled forecast for a set of panchayats (vectorised: one model pass per request)."""
    block_fc = pd.DataFrame([fetch_block_forecast(b, month) for b in rows.block], index=rows.index)
    features = rows.assign(month=month, **block_fc)[bundle["features"]]

    block_rain = block_fc.block_forecast_rain_mm.to_numpy()
    rain = np.clip(block_rain + rain_model.predict(features), 0, None)
    temp = block_fc.block_forecast_temp_c.to_numpy() + temp_model.predict(features[["elevation_m"]])
    humidity = float(bundle["humidity_by_month"][month])
    wind = float(bundle["wind_mean"])

    # Chance of rain: share of the forest's trees that predict at least 2.5 mm
    X = rain_model.named_steps["prep"].transform(features)
    per_tree = np.stack([t.predict(X) for t in rain_model.named_steps["reg"].estimators_]) + block_rain
    chance = (100 * (per_tree >= RAIN_DAY_MM).mean(axis=0)).round().astype(int)

    out = []
    for i, (_, row) in enumerate(rows.iterrows()):
        r, t = round(float(rain[i]), 1), round(float(temp[i]), 1)
        out.append({
            "panchayat_id": row.panchayat_id,
            "panchayat": row.panchayat,
            "block": row.block,
            "district": row.district,
            "lat": float(row.lat),
            "lon": float(row.lon),
            "block_forecast_rain_mm": float(block_rain[i]),
            "rainfall_mm": r,
            "rain_category": rain_category(r),
            "temperature_c": t,
            "humidity_pct": round(humidity, 1),
            "wind_kmph": round(wind, 1),
            "rain_probability_pct": int(chance[i]),
            "hist_avg_rainfall_mm": float(row.hist_avg_rainfall_mm),
            "advisory": advisory_for(r, t, humidity),
        })
    return out


MonthQuery = Query(None, ge=1, le=12, description="1-12; defaults to the current month")


@app.get("/districts")
def list_districts():
    return sorted(master.district.unique().tolist())


@app.get("/blocks/{district}")
def list_blocks(district: str):
    blocks = master[master.district == district].block.unique().tolist()
    if not blocks:
        raise HTTPException(404, "District not found")
    return sorted(blocks)


@app.get("/panchayats/{block}")
def list_panchayats(block: str):
    rows = master[master.block == block]
    if rows.empty:
        raise HTTPException(404, "Block not found")
    return rows[["panchayat_id", "panchayat"]].to_dict("records")


@app.get("/forecast/{panchayat_id}")
def forecast(panchayat_id: str, month: int | None = MonthQuery):
    month = month or datetime.now().month
    row = master[master.panchayat_id == panchayat_id]
    if row.empty:
        raise HTTPException(404, "Panchayat not found")
    return predict_rows(row, month)[0]


@app.get("/forecast/block/{block}")
def forecast_block(block: str, month: int | None = MonthQuery):
    month = month or datetime.now().month
    rows = master[master.block == block]
    if rows.empty:
        raise HTTPException(404, "Block not found")
    return predict_rows(rows, month)


@app.get("/model/skill")
def model_skill():
    report = BASE / "validation_report.json"
    if not report.exists():
        raise HTTPException(404, "Run train_model.py to create validation_report.json")
    return json.loads(report.read_text())
