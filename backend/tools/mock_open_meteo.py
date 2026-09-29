"""
mock_open_meteo.py — OFFLINE TEST DOUBLE for the Open-Meteo APIs.

Only used when the environment variable AEROATMOS_MOCK=1 is set. It fabricates
plausible Bihar-like weather so the whole pipeline (fetch -> train -> API ->
dashboard) can be tested without internet. Nothing produced in mock mode is real
data: fetch_real_data.py records source="mock" and the API/dashboard show a
"SIMULATED DATA" banner. Never present mock results as real.

The simulator has a shared regional weather signal, local panchayat-scale
variation (elevation, a smooth spatial rain pattern), and a coarse "GFS" block
forecast whose errors grow with lead time, so a downscaling model has something
to learn.
"""

import zlib
from functools import lru_cache

import numpy as np
import pandas as pd

# Monthly climatology loosely based on central Bihar (Patna/Gaya)
TMAX = [23, 26, 32, 37, 38, 36, 32, 32, 32, 31, 28, 24]
TMIN = [9, 12, 17, 22, 26, 27, 26, 26, 25, 21, 14, 10]
RAIN_P = [.06, .06, .05, .06, .12, .40, .60, .55, .45, .12, .03, .03]   # chance of a rain day
RAIN_MEAN = [6, 6, 5, 6, 9, 14, 16, 15, 14, 12, 5, 5]                  # mm on a rain day
RH = [72, 62, 48, 42, 52, 72, 84, 85, 83, 76, 70, 73]
WIND = [5, 7, 10, 12, 11, 10, 8, 7, 6, 5, 4, 4]
WDIR = [290, 280, 275, 270, 180, 110, 100, 100, 100, 300, 300, 290]
CLOUD = [15, 15, 15, 20, 35, 65, 85, 82, 70, 30, 10, 12]


def _seed(*parts) -> int:
    """Stable across processes (unlike hash())."""
    return zlib.crc32("|".join(f"{p:.3f}" if isinstance(p, float) else str(p) for p in parts).encode())


def elevation(lat, lon):
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    base = 52 + np.clip(25.35 - lat, 0, None) * 140                           # rises towards Gaya
    hills = 180 * np.exp(-((lat - 24.62) ** 2 + (lon - 84.88) ** 2) / 0.004)  # hills near Sherghati
    ripple = 6 * np.sin(lat * 90) * np.cos(lon * 70)
    return base + hills + ripple


@lru_cache(maxsize=64)
def _regional_range(start_ord: int, end_ord: int) -> pd.DataFrame:
    """Shared day-to-day weather for two sub-regions (north = Patna, south = Gaya)."""
    rows, idx = [], []
    for o in range(start_ord, end_ord + 1):
        d = pd.Timestamp.fromordinal(o)
        r = np.random.default_rng(o)
        m = d.month - 1
        wet_n = r.random() < RAIN_P[m]
        wet_s = (r.random() < 0.7) if wet_n else (r.random() < RAIN_P[m] * 0.6)
        wet = wet_n or wet_s
        rows.append((r.gamma(0.9, RAIN_MEAN[m] / 0.9) * wet_n, r.gamma(0.9, RAIN_MEAN[m] / 0.9) * wet_s,
                     TMAX[m] + r.normal(0, 1.8) - 2.5 * wet, TMIN[m] + r.normal(0, 1.4),
                     RH[m] + r.normal(0, 5) + 8 * wet, max(WIND[m] + r.normal(0, 2), 1),
                     WDIR[m] + r.normal(0, 25), float(np.clip(CLOUD[m] + r.normal(0, 15) + 25 * wet, 0, 100))))
        idx.append(d)
    return pd.DataFrame(rows, index=pd.DatetimeIndex(idx),
                        columns=["rain_n", "rain_s", "tmax", "tmin", "rh", "wind", "wdir", "cloud"])


def _regional(dates: pd.DatetimeIndex) -> pd.DataFrame:
    return _regional_range(dates[0].toordinal(), dates[-1].toordinal()).loc[dates]


def local_daily(lat: float, lon: float, dates: pd.DatetimeIndex, noise: float = 1.0) -> pd.DataFrame:
    """'True' daily weather at a point: regional signal + local, learnable variation."""
    reg = _regional(dates)
    w_south = float(np.clip((25.3 - lat) / 0.5, 0, 1))
    rain = (1 - w_south) * reg.rain_n + w_south * reg.rain_s
    elev = float(elevation(lat, lon))
    rain_mult = np.exp(0.35 * np.sin(lat * 60) * np.cos(lon * 45) + 0.0015 * (elev - 60))
    r = np.random.default_rng(_seed("local", lat, lon, dates[0].toordinal()))
    eps = r.normal(0, 1, size=(len(dates), 4)) * noise
    df = pd.DataFrame(index=dates)
    df["rain"] = np.clip(rain * rain_mult * np.exp(0.25 * eps[:, 0]), 0, None)
    lapse = -0.0065 * (elev - 55)
    df["tmax"] = reg.tmax + lapse + 0.4 * np.cos(lon * 50) + 0.3 * eps[:, 1]
    df["tmin"] = reg.tmin + lapse - 0.6 * np.sin(lat * 40) + 0.3 * eps[:, 2]
    df["rh"] = np.clip(reg.rh - 0.02 * (elev - 55) + 2.5 * np.sin(lon * 60) + 1.5 * eps[:, 3], 15, 100)
    df["wind"] = reg.wind * (1 + 0.002 * (elev - 55))
    df["wdir"] = reg.wdir
    df["cloud"] = reg.cloud
    return df


def gfs_daily(lat: float, lon: float, dates: pd.DatetimeIndex, lead: int) -> pd.DataFrame:
    """Coarse (~25 km) forecast with lead-dependent errors."""
    pts = [(lat + a, lon + b) for a in (-0.12, 0, 0.12) for b in (-0.12, 0, 0.12)]
    coarse = sum(local_daily(a, b, dates, noise=0) for a, b in pts) / len(pts)
    r = np.random.default_rng(_seed("gfs", lat, lon, lead, dates[0].toordinal()))
    n = len(dates)
    f = coarse.copy()
    miss = r.random(n) < 0.08 * lead
    false = (r.random(n) < 0.04 * lead) & (coarse.rain < 1)
    f["rain"] = np.where(miss, 0, coarse.rain * np.exp(r.normal(0.1, 0.35 + 0.12 * lead, n)))
    f["rain"] = np.where(false, r.gamma(0.8, 6, n), f.rain)
    f["tmax"] = coarse.tmax + 1.2 + r.normal(0, 0.8 + 0.3 * lead, n)
    f["tmin"] = coarse.tmin - 0.6 + r.normal(0, 0.7 + 0.25 * lead, n)
    f["rh"] = np.clip(coarse.rh - 6 + r.normal(0, 4 + lead, n), 10, 100)
    f["wind"] = coarse.wind + 2.5 + r.normal(0, 1 + 0.3 * lead, n)
    f["wdir"] = coarse.wdir + r.normal(0, 15 * lead, n)
    f["cloud"] = np.clip(coarse.cloud + r.normal(0, 10 + 3 * lead, n), 0, 100)
    return f


def to_hourly(daily: pd.DataFrame) -> pd.DataFrame:
    hours = np.arange(24)
    temp_shape = np.tile((np.cos((hours - 14) / 24 * 2 * np.pi) + 1) / 2, len(daily))  # 1 at 14:00
    rain_shape = np.zeros(24); rain_shape[[15, 16, 17, 18]] = [.2, .4, .3, .1]
    rep = lambda col: np.repeat(daily[col].to_numpy(float), 24)
    idx = pd.DatetimeIndex(np.repeat(daily.index.values, 24)) + pd.to_timedelta(np.tile(hours, len(daily)), unit="h")
    tmin, tmax = rep("tmin"), rep("tmax")
    return pd.DataFrame({
        "precipitation": rep("rain") * np.tile(rain_shape, len(daily)),
        "temperature_2m": tmin + (tmax - tmin) * temp_shape,
        "relative_humidity_2m": np.clip(rep("rh") + 12 * (0.5 - temp_shape), 5, 100),
        "wind_speed_10m": rep("wind") * (0.8 + 0.4 * temp_shape),
        "wind_direction_10m": rep("wdir") % 360,
        "cloud_cover": rep("cloud"),
    }, index=idx)


def _payload(lat, lon, hourly: pd.DataFrame, wanted: list[str]) -> dict:
    return {"latitude": lat, "longitude": lon, "timezone": "Asia/Kolkata",
            "hourly": {"time": hourly.index.strftime("%Y-%m-%dT%H:%M").tolist(),
                       **{v: np.round(hourly[v].to_numpy(float), 2).tolist() for v in wanted}}}


def respond(url: str, params: dict):
    lats = [float(x) for x in str(params["latitude"]).split(",")]
    lons = [float(x) for x in str(params["longitude"]).split(",")]
    if url.endswith("/elevation"):
        return {"elevation": [round(float(e), 1) for e in elevation(lats, lons)]}

    wanted = params["hourly"].split(",")
    if "start_date" in params:
        dates = pd.date_range(params["start_date"], params["end_date"], freq="D")
    else:
        today = pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None).normalize()
        dates = pd.date_range(today, periods=int(params.get("forecast_days", 7)), freq="D")

    results = []
    for lat, lon in zip(lats, lons):
        if "archive" in url:  # ERA5-Land-like: values for the 0.1 degree grid cell
            glat, glon = round(lat, 1), round(lon, 1)
            results.append(_payload(glat, glon, to_hourly(local_daily(glat, glon, dates)), wanted))
        elif "previous-runs" in url:
            h = pd.concat([to_hourly(gfs_daily(lat, lon, dates, lead)).add_suffix(f"_previous_day{lead}")
                           for lead in range(1, 6)], axis=1)
            results.append(_payload(lat, lon, h, wanted))
        else:  # live forecast issued today: day i has lead max(i, 1)
            h = pd.concat([to_hourly(gfs_daily(lat, lon, dates[i:i + 1], max(i, 1))) for i in range(len(dates))])
            results.append(_payload(lat, lon, h, wanted))
    return results[0] if len(results) == 1 else results
