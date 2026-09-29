"""
app.py — AeroAtmos backend: 5-day, panchayat-level forecasts downscaled from
block forecasts, with IMD-style farmer advisories.

Endpoints
---------
GET  /districts
GET  /blocks/{district}
GET  /panchayats/{block}
GET  /forecast/{panchayat_id}     5-day forecast + advisories for one panchayat
GET  /forecast/block/{block}      5-day forecast for every panchayat in a block
GET  /geo/block/{block}           panchayat boundaries (GeoJSON) for the map
GET  /model/skill                 validation report written by train_model.py
GET  /meta                        where the data comes from and when it was issued

Run after fetch_real_data.py and train_model.py, from the repo root:
    uvicorn app:app --reload --app-dir backend
Then open http://127.0.0.1:8000/docs
"""

import json
import threading
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import features as F
import weather as W

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
LIVE_FILE = DATA / "live_block_forecast.json"
LIVE_TTL_S = 3600

app = FastAPI(title="AeroAtmos API — panchayat-level weather downscaling", version="0.3.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

gps, blocks = F.load_places(DATA)
GEO = json.loads((DATA / "panchayats.geojson").read_text())
META = json.loads((DATA / "fetch_meta.json").read_text()) if (DATA / "fetch_meta.json").exists() else {}
MODEL_FILE = BASE / "models" / "downscaling.joblib"
bundle = joblib.load(MODEL_FILE) if MODEL_FILE.exists() else None

# IMD terminology for 24-hour rainfall: (upper limit in mm, name)
IMD_RAIN = [(0.1, "No rain"), (2.5, "Very light rain"), (15.6, "Light rain"), (64.5, "Moderate rain"),
            (115.6, "Heavy rain"), (204.5, "Very heavy rain"), (float("inf"), "Extremely heavy rain")]


def rain_category(mm: float) -> str:
    return next(name for upper, name in IMD_RAIN if mm < upper)


def day_advice(d: dict) -> list[str]:
    rain, tmax, tmin, rh, wind = d["rainfall_mm"], d["tmax_c"], d["tmin_c"], d["humidity_pct"], d["wind_kmph"]
    cat = d["rain_category"]
    tips = []
    if rain >= 64.5:
        tips += [f"{cat} — do not irrigate; clear field drainage",
                 "Postpone fertilizer and pesticide application; protect harvested produce"]
    elif rain >= 15.6:
        tips += [f"{cat} — skip irrigation", "Hold off on spraying; it would wash off"]
    elif rain >= 2.5:
        tips += [f"{cat} — reduce irrigation", "Spray only in dry spells"]
    elif d["rain_chance_pct"] < 30 and wind < 15:
        tips.append("Dry and calm — a good window for spraying and field work")
    if tmax >= 40:
        tips.append("Very hot — irrigate in the early morning or evening; shade and water for livestock")
    elif tmax >= 36 and rain < 2.5:
        tips.append("Hot and dry — avoid spraying in the afternoon heat")
    if tmin <= 4:
        tips.append("Frost risk — light irrigation in the evening protects crops")
    elif tmin <= 10:
        tips.append("Cold night — protect nurseries and young livestock")
    if rh >= 85 and tmin >= 18:
        tips.append("Humid and warm — watch crops for fungal disease")
    if wind >= 25:
        tips.append("Strong wind — avoid spraying; support tall crops")
    return tips or ["No weather risk expected — routine field operations"]


def outlook(days: list[dict]) -> str:
    wet = [d for d in days[:3] if d["rainfall_mm"] >= 15.6]
    if wet:
        d = max(wet, key=lambda x: x["rainfall_mm"])
        return f"{d['rain_category']} likely on {d['weekday']} — plan irrigation and spraying around it."
    if sum(d["rainfall_mm"] for d in days) < 2.5 and max(d["tmax_c"] for d in days) >= 35:
        return "Dry, hot spell for the next 5 days — schedule irrigation."
    return "No major rain event expected in the next 3 days."


_lock = threading.Lock()
_cache = {"at": 0.0, "live": None, "pred": None}


def live_block_forecast() -> dict:
    try:
        data = W.fetch_live_block_forecast(blocks, META.get("forecast_model", W.FORECAST_MODEL),
                                           retries=1, timeout=10, pause=0)
        data["status"] = "live"
        LIVE_FILE.write_text(json.dumps(data, indent=1))
    except Exception as exc:  # noqa: BLE001 - fall back to the last saved forecast
        if not LIVE_FILE.exists():
            raise HTTPException(503, f"No live forecast ({exc}) and no saved snapshot; run fetch_real_data.py")
        data = json.loads(LIVE_FILE.read_text())
        data["status"] = "saved snapshot (could not reach Open-Meteo)"
    return data


def predictions() -> tuple[dict, pd.DataFrame]:
    """Downscale the current block forecast to every panchayat (cached for an hour)."""
    if bundle is None:
        raise HTTPException(503, "Model not trained yet: run fetch_real_data.py, then train_model.py")
    with _lock:
        if _cache["pred"] is not None and time.time() - _cache["at"] < LIVE_TTL_S:
            return _cache["live"], _cache["pred"]
        live = live_block_forecast()
        bf = pd.DataFrame([{**d, "block_lgd": int(b)} for b, days in live["blocks"].items() for d in days])
        X = F.build(bf, gps, blocks)
        feats = bundle["features"]
        for v, model in bundle["models"].items():
            lo, hi = bundle["limits"].get(v, (None, None))
            pred = X[f"bf_{v}"] + model.predict(X[feats])
            X[v] = pred.clip(lower=lo, upper=hi) if (lo is not None or hi is not None) else pred
        X["rain_chance"] = bundle["rain_chance"].predict_proba(X[feats])[:, 1]
        _cache.update(at=time.time(), live=live, pred=X)
        return live, X


def r1(x):
    return None if x is None or pd.isna(x) else round(float(x), 1)


def panchayat_forecast(gp: pd.Series, X: pd.DataFrame) -> dict:
    rows = X[X.panchayat_id == gp.panchayat_id].sort_values("lead_day")
    days = []
    for _, r in rows.iterrows():
        d = {"date": r.date.strftime("%Y-%m-%d"), "weekday": r.date.strftime("%a"), "lead_day": int(r.lead_day),
             "block_forecast": {"rainfall_mm": r1(r.bf_rain), "tmax_c": r1(r.bf_tmax), "tmin_c": r1(r.bf_tmin),
                                "humidity_pct": r1(r.bf_rh), "wind_kmph": r1(r.bf_wind)},
             "rainfall_mm": r1(r.rain), "rain_chance_pct": int(round(100 * r.rain_chance)),
             "tmax_c": r1(r.tmax), "tmin_c": r1(r.tmin), "humidity_pct": r1(r.rh), "wind_kmph": r1(r.wind),
             "wind_dir": W.compass(r.bf_wdir_sin, r.bf_wdir_cos),
             "cloud_octa": None if pd.isna(r.bf_cloud) else int(round(r.bf_cloud / 12.5))}
        d["rain_category"] = rain_category(d["rainfall_mm"])
        d["advisory"] = day_advice(d)
        days.append(d)
    return {"panchayat_id": gp.panchayat_id, "panchayat": gp.panchayat, "block": gp.block,
            "district": gp.district, "lat": float(gp.lat), "lon": float(gp.lon),
            "area_km2": float(gp.area_km2), "days": days, "outlook": outlook(days) if days else ""}


@app.get("/districts")
def list_districts():
    return sorted(gps.district.unique().tolist())


@app.get("/blocks/{district}")
def list_blocks(district: str):
    b = sorted(gps[gps.district == district].block.unique().tolist())
    if not b:
        raise HTTPException(404, "District not found")
    return b


@app.get("/panchayats/{block}")
def list_panchayats(block: str):
    rows = gps[gps.block == block]
    if rows.empty:
        raise HTTPException(404, "Block not found")
    return rows[["panchayat_id", "panchayat"]].to_dict("records")


@app.get("/forecast/{panchayat_id}")
def forecast(panchayat_id: str):
    row = gps[gps.panchayat_id == panchayat_id]
    if row.empty:
        raise HTTPException(404, "Panchayat not found")
    live, X = predictions()
    return {**panchayat_forecast(row.iloc[0], X), "issued_at": live["issued_at"], "status": live["status"]}


@app.get("/forecast/block/{block}")
def forecast_block(block: str):
    rows = gps[gps.block == block]
    if rows.empty:
        raise HTTPException(404, "Block not found")
    live, X = predictions()
    return {"block": block, "issued_at": live["issued_at"], "status": live["status"],
            "panchayats": [panchayat_forecast(r, X) for _, r in rows.iterrows()]}


@app.get("/geo/block/{block}")
def geo_block(block: str):
    feats = [f for f in GEO["features"] if f["properties"]["block"] == block]
    if not feats:
        raise HTTPException(404, "Block not found")
    return {"type": "FeatureCollection", "features": feats}


@app.get("/model/skill")
def model_skill():
    report = BASE / "validation_report.json"
    if not report.exists():
        raise HTTPException(404, "Run train_model.py to create validation_report.json")
    return json.loads(report.read_text())


@app.get("/meta")
def meta():
    return {"data_source": META.get("source", "not downloaded yet"),
            "simulated": META.get("source") == "mock",
            "block_forecast_model": META.get("forecast_model", W.FORECAST_MODEL),
            "truth_model": META.get("truth_model", W.TRUTH_MODEL),
            "history": f"{META.get('start')} to {META.get('end')}" if META else None,
            "fetched_at": META.get("fetched_at"),
            "panchayats": int(len(gps)), "blocks": int(len(blocks)),
            "boundaries": "LGD village boundaries dissolved to gram panchayats (datameet / ramSeraph)"}
