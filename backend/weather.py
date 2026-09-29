"""
weather.py — shared helpers for talking to Open-Meteo and turning hourly
weather into the daily values IMD's agromet forecasts use.

Used by fetch_real_data.py (history for training) and app.py (live forecast).

Open-Meteo (https://open-meteo.com) is free for non-commercial use, needs no
API key, and exposes archived forecasts ("Previous Runs"), reanalysis
("Archive") and live forecasts through the same JSON format.
"""

import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

TZ = "Asia/Kolkata"

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
ELEVATION_URL = "https://api.open-meteo.com/v1/elevation"

# Block forecast input: NOAA GFS, a stand-in for IMD's GFS-based block forecasts.
FORECAST_MODEL = "gfs_seamless"
# "Truth" for training: ECMWF IFS 9 km analysis archive (has rainfall at native resolution;
# ERA5-Land's rainfall comes from 0.25-degree ERA5). Replace with station data when available.
TRUTH_MODEL = "ecmwf_ifs"

FORECAST_HOURLY = ["precipitation", "temperature_2m", "relative_humidity_2m",
                   "wind_speed_10m", "wind_direction_10m", "cloud_cover"]
TRUTH_HOURLY = ["precipitation", "temperature_2m", "relative_humidity_2m", "wind_speed_10m"]

# Daily variables: IMD block forecasts give rainfall, max/min temperature,
# humidity, wind speed and direction, and cloud cover.
DAILY = ["rain", "tmax", "tmin", "rh", "wind", "wdir_sin", "wdir_cos", "cloud"]
TARGETS = ["rain", "tmax", "tmin", "rh", "wind"]  # what gets downscaled
LEAD_DAYS = [1, 2, 3, 4, 5]

MOCK = os.environ.get("AEROATMOS_MOCK") == "1"  # offline testing only, see tools/mock_open_meteo.py


class OpenMeteoError(RuntimeError):
    pass


def get_json(url: str, params: dict, cache_dir: Path | None = None, retries: int = 6,
             pause: float = 1.0, timeout: float = 90):
    """GET with retries, polite pacing and an optional on-disk cache (so re-runs resume)."""
    key = hashlib.sha1((url + json.dumps(params, sort_keys=True)).encode()).hexdigest()
    cached = cache_dir / f"{key}.json" if cache_dir else None
    if cached and cached.exists():
        return json.loads(cached.read_text())

    if MOCK:
        from tools.mock_open_meteo import respond
        data = respond(url, params)
    else:
        import requests
        wait = 5.0
        for attempt in range(retries):
            try:
                r = requests.get(url, params=params, timeout=timeout)
            except requests.RequestException as exc:
                if attempt == retries - 1:
                    raise OpenMeteoError(f"network error: {exc}") from exc
                time.sleep(wait); wait *= 2
                continue
            if r.status_code == 200:
                data = r.json()
                break
            reason = _reason(r)
            if r.status_code == 429 or r.status_code >= 500:
                if attempt == retries - 1:
                    raise OpenMeteoError(f"HTTP {r.status_code}: {reason}")
                time.sleep(60 if r.status_code == 429 else wait); wait *= 2
                continue
            raise OpenMeteoError(f"HTTP {r.status_code}: {reason}  (url={r.url})")
        time.sleep(pause)

    if cached:
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_text(json.dumps(data))
    return data


def _reason(r) -> str:
    try:
        return r.json().get("reason", r.text[:300])
    except Exception:
        return r.text[:300]


def as_list(resp) -> list[dict]:
    """Open-Meteo returns a dict for one location and a list for several."""
    return resp if isinstance(resp, list) else [resp]


def hourly_frame(resp: dict) -> pd.DataFrame:
    h = resp["hourly"]
    df = pd.DataFrame({k: v for k, v in h.items() if k != "time"})
    df.index = pd.to_datetime(h["time"])
    return df.apply(pd.to_numeric, errors="coerce")


def daily_from_hourly(h: pd.DataFrame, min_hours: int = 20) -> pd.DataFrame:
    """Aggregate hourly Open-Meteo variables (local time) to IMD-style daily values."""
    h = h.copy()
    h["date"] = h.index.normalize()
    g = h.groupby("date")
    out = pd.DataFrame(index=g.size().index)
    count = g.size()
    if "precipitation" in h:
        out["rain"] = g["precipitation"].sum(min_count=min_hours)
    if "temperature_2m" in h:
        out["tmax"] = g["temperature_2m"].max()
        out["tmin"] = g["temperature_2m"].min()
    if "relative_humidity_2m" in h:
        out["rh"] = g["relative_humidity_2m"].mean()
    if "wind_speed_10m" in h:
        out["wind"] = g["wind_speed_10m"].mean()
    if "wind_direction_10m" in h and "wind_speed_10m" in h:
        # speed-weighted vector mean of the direction the wind blows FROM
        rad = np.radians(h["wind_direction_10m"])
        h["_u"] = h["wind_speed_10m"] * np.sin(rad)
        h["_v"] = h["wind_speed_10m"] * np.cos(rad)
        u, v = g["_u"].mean(), g["_v"].mean()
        norm = np.hypot(u, v).replace(0, np.nan)
        out["wdir_sin"], out["wdir_cos"] = u / norm, v / norm
    if "cloud_cover" in h:
        out["cloud"] = g["cloud_cover"].mean()
    out = out[count >= min_hours]  # drop partial days
    out.index.name = "date"
    return out


def compass(sin: float, cos: float) -> str:
    if sin is None or cos is None or (isinstance(sin, float) and math.isnan(sin)):
        return "-"
    deg = (math.degrees(math.atan2(sin, cos)) + 360) % 360
    return ["N", "NE", "E", "SE", "S", "SW", "W", "NW"][int((deg + 22.5) // 45) % 8]


def previous_runs_params(lat: float, lon: float, start: str, end: str, model: str) -> dict:
    hourly = [f"{v}_previous_day{d}" for d in LEAD_DAYS for v in FORECAST_HOURLY]
    return {"latitude": lat, "longitude": lon, "hourly": ",".join(hourly), "models": model,
            "start_date": start, "end_date": end, "timezone": TZ}


def split_leads(h: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """Previous Runs columns 'temperature_2m_previous_day3' -> {3: frame with 'temperature_2m'}."""
    out = {}
    for d in LEAD_DAYS:
        suffix = f"_previous_day{d}"
        cols = {c: c[: -len(suffix)] for c in h.columns if c.endswith(suffix)}
        out[d] = h[list(cols)].rename(columns=cols)
    return out


def fetch_live_block_forecast(blocks: pd.DataFrame, model: str = FORECAST_MODEL,
                              today: pd.Timestamp | None = None, **http) -> dict:
    """Current forecast for each block centroid -> next 5 days of daily values (lead 1..5)."""
    resp = as_list(get_json(FORECAST_URL, {
        "latitude": ",".join(str(x) for x in blocks.lat),
        "longitude": ",".join(str(x) for x in blocks.lon),
        "hourly": ",".join(FORECAST_HOURLY), "models": model, "timezone": TZ,
        "forecast_days": 7}, **http))
    today = (today or pd.Timestamp.now(tz=TZ).tz_localize(None)).normalize()
    result = {}
    for (_, b), r in zip(blocks.iterrows(), resp):
        daily = daily_from_hourly(hourly_frame(r))
        days = []
        for d in LEAD_DAYS:
            date = today + pd.Timedelta(days=d)
            if date in daily.index:
                row = daily.loc[date]
                days.append({"date": date.strftime("%Y-%m-%d"), "lead_day": d,
                             **{k: (None if pd.isna(row.get(k)) else round(float(row[k]), 3)) for k in DAILY}})
        result[str(b.block_lgd)] = days
    return {"issued_at": pd.Timestamp.now(tz=TZ).strftime("%Y-%m-%d %H:%M IST"),
            "model": model, "source": "mock" if MOCK else "open-meteo", "blocks": result}
