# AeroAtmos
### AI-Powered Hyperlocal Weather Downscaling

**Team AERO VISION** · Smart India Hackathon 2026 — Problem Statement **SIH 26074** (Ministry of Earth Sciences / IMD)

---

## The Problem

IMD (India Meteorological Department) publishes weather forecasts at the **Block** level. But a farmer makes decisions — irrigate today, delay spraying, protect a harvest — at the **Panchayat** level. Rainfall and temperature can vary meaningfully within a single block due to terrain, land use and local weather, so a block-wide forecast can mislead a farmer in a specific village.

## Our Solution

AeroAtmos takes a block-level forecast and uses machine learning to **downscale** it to every gram panchayat in the block, for the next **5 days** — the horizon of IMD's agromet advisories — and turns it into a **plain-language farmer advisory**.

- **Real panchayats:** 87 gram panchayats in 5 blocks of Patna and Gaya, with official LGD codes and boundaries.
- **Real data:** archived GFS block forecasts and ECMWF analyses since Jan 2024, via Open-Meteo.
- **IMD's variables:** rainfall, max/min temperature, humidity, wind speed and direction, cloud cover.
- **Proof, not claims:** every trained model is scored on panchayats and dates it never saw, against the block forecast and two classical baselines.

---

## What's in this repo

```
AeroAtmos/
├── backend/
│   ├── fetch_real_data.py     # Downloads real training data (run once, needs internet)
│   ├── train_model.py         # Trains the models and validates them against baselines
│   ├── app.py                 # FastAPI backend: live 5-day panchayat forecasts + advisories
│   ├── features.py            # Model inputs (shared by training and the API)
│   ├── weather.py             # Open-Meteo client + hourly-to-daily (IMD-style) aggregation
│   ├── requirements.txt       # Pinned versions (Python 3.11+)
│   ├── data/
│   │   ├── blocks.csv            # 5 blocks: LGD code, centroid, area
│   │   ├── panchayats.csv        # 87 gram panchayats: LGD code, block, centroid, area
│   │   ├── panchayats.geojson    # Panchayat boundaries for the map
│   │   └── (created by fetch_real_data.py: static_features.csv, block_forecasts.csv,
│   │        panchayat_obs.csv, live_block_forecast.json, fetch_meta.json)
│   ├── models/                # Created by train_model.py
│   └── tools/
│       ├── build_panchayats.py   # Rebuilds data/panchayats.* from LGD boundaries (already done)
│       └── mock_open_meteo.py    # Offline test double — simulated data, never present as real
├── weather-down-live.html     # Dashboard: map, 5-day timeline, charts, advisory, model skill
└── docs/                      # SIH 2026 idea presentation (PPTX + PDF)
```

## Data sources

| What | Source | Notes |
|---|---|---|
| Panchayat list and boundaries | LGD village boundaries (LGD/Bharatmaps, via [datameet / ramSeraph](https://github.com/ramSeraph/indian_admin_boundaries)) | Villages dissolved into gram panchayats |
| Block forecast (model input) | NOAA GFS, archived runs 1–5 days ahead, via [Open-Meteo Previous Runs API](https://open-meteo.com/en/docs/previous-runs-api) | Stand-in for IMD's block forecast feed, which is not public |
| Panchayat weather (training truth) | ECMWF IFS 9 km analysis archive via [Open-Meteo](https://open-meteo.com/en/docs/historical-weather-api) | ~9 km grid: coarser than a panchayat, see limitations |
| Elevation, terrain relief | Copernicus DEM GLO-90 via [Open-Meteo Elevation API](https://open-meteo.com/en/docs/elevation-api) | |
| Land cover (optional) | ESA WorldCover 10 m | Needs `pip install rasterio` |
| Live 5-day forecast | NOAA GFS via Open-Meteo Forecast API | Refreshed hourly by the API; a saved copy is used offline |

Open-Meteo is free for non-commercial use and needs no API key.

---

## Getting started

Requires **Python 3.11+**. Run these from the repo root.

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux
pip install -r backend/requirements.txt
```

**1. Download the real data** (once; ~10–20 minutes; resumes if interrupted)

```bash
cd backend
python fetch_real_data.py --check     # 10-second connectivity test
python fetch_real_data.py
```

**2. Train and validate** (a few minutes)

```bash
python train_model.py
```

It prints a table of errors for the model and each baseline, and writes `validation_report.json`.

**3. Start the API**

```bash
uvicorn app:app --reload        # from backend/  (or: uvicorn app:app --app-dir backend  from the root)
```

API docs: http://127.0.0.1:8000/docs

**4. Open the dashboard**

Open `weather-down-live.html` in a browser while the API is running. If the API runs elsewhere: `weather-down-live.html?api=https://your-server.example.com`.

> Commit the files from steps 1–2 (`backend/data/*.csv`, `*.json`, `backend/models/`) so teammates can skip straight to step 3. Don't commit `backend/data/cache/`.

**If the download fails:** run `--check` to see which service fails. Re-running resumes from the cache. If Open-Meteo rejects the forecast model, try `--forecast-model gfs_global` (or `ecmwf_ifs025`); if it reports a rate limit, wait an hour and re-run.

---

## How the model works

For each variable (rainfall, max/min temperature, humidity, wind) a gradient-boosted model (scikit-learn `HistGradientBoostingRegressor`) learns the **correction** to the block forecast for a panchayat — *panchayat value minus block forecast* — from:

- the block forecast itself, at lead day 1–5
- block forecasts interpolated from neighbouring blocks (distance-weighted)
- the panchayat's position relative to its block centre, its size, elevation and terrain relief (and land cover, if fetched)
- the season

Predicting the correction keeps every forecast anchored to the block forecast. A classifier gives the **chance of rain** (≥ 2.5 mm, IMD's "light rain" threshold).

Advisories use IMD's 24-hour rainfall categories (very light 0.1–2.4 mm, light 2.5–15.5, moderate 15.6–64.4, heavy 64.5–115.5, very heavy 115.6–204.4, extremely heavy ≥ 204.5) plus heat, frost, humidity and wind rules, and a 5-day outlook.

## Validation

`train_model.py` holds out the **last 180 days** and **a quarter of the panchayats in every block** (never used in training), and compares the model on those unseen panchayats with:

1. the **block forecast as-is** — what a farmer gets today
2. the **bias-corrected block forecast** — average correction per lead day and month
3. **interpolation between block forecasts** — classical spatial downscaling

Reported: mean absolute error per variable and lead day; for rain, hit rate / false-alarm ratio / critical success index at 2.5 mm and 15.6 mm; and the Brier score of the chance of rain. The dashboard shows the same table ("How much better than the block forecast?").

**Results on real data** — 22 of 87 panchayats never seen in training, test period 2026-03-26 – 2026-09-21, truth = ECMWF IFS 9 km analysis. Mean absolute error:

| Variable | Our model | Block forecast as-is | Bias-corrected block | Interpolated blocks | Error vs block forecast |
|---|---|---|---|---|---|
| Rainfall (mm) | 3.82 | 4.35 | 4.69 | 4.31 | −12% |
| Max temperature (°C) | 1.62 | 4.37 | 2.19 | 4.36 | −63% |
| Min temperature (°C) | 1.07 | 2.47 | 1.38 | 2.46 | −57% |
| Humidity (%) | 6.54 | 19.62 | 8.02 | 19.59 | −67% |
| Wind (km/h) | 2.53 | 5.72 | 2.94 | 5.71 | −56% |

- Rain days (≥ 2.5 mm) caught: 81% vs 64% for the block forecast (false alarms 31% vs 22%).
- Known weak spots: moderate-rain days (≥ 15.6 mm) caught 32% vs 45% for the block forecast; the chance-of-rain Brier score is 0.178 vs 0.158 for the monthly average, so it is not yet better than climatology.

---

## API Endpoints

| Endpoint | Description |
|---|---|
| `GET /districts`, `/blocks/{district}`, `/panchayats/{block}` | Lists |
| `GET /forecast/{panchayat_id}` | 5-day forecast + advisories + outlook for one panchayat |
| `GET /forecast/block/{block}` | 5-day forecast for every panchayat in a block |
| `GET /geo/block/{block}` | Panchayat boundaries (GeoJSON) |
| `GET /model/skill` | Validation report |
| `GET /meta` | Data provenance: sources, history period, whether data is simulated |

---

## Limitations & next steps

- **Training truth is a ~9 km model analysis, not panchayat observations.** It is real data and good for building and testing the pipeline, but panchayats are ~3–4 km across. Next step: train and validate against station data — IMD automatic weather stations and Bihar Mausam Sewa Kendra's block weather stations and panchayat rain gauges.
- **GFS stands in for IMD's block forecast.** The pipeline takes any block forecast; swap in IMD's feed when available.
- **Soil** is not yet a feature (SoilGrids is the planned source).
- **Delivery:** Hindi/local-language advisories and SMS/IVR/WhatsApp alerts for farmers without smartphones.

## Offline test mode

`AEROATMOS_MOCK=1 python fetch_real_data.py` fabricates plausible weather (tools/mock_open_meteo.py) so the pipeline can be tested without internet. Everything produced this way is marked `source: mock`, and the dashboard shows a **SIMULATED DATA** banner. Never present these results as real.

---

## Team AERO VISION

Built for Smart India Hackathon 2026 — Problem Statement 26074.
