# AeroAtmos
### AI-Powered Hyperlocal Weather Downscaling

**Team AERO VISION** · Smart India Hackathon — Problem Statement **SIH 26074**

---

## The Problem

IMD (India Meteorological Department) publishes weather forecasts at the **Block** level. But a farmer makes decisions — irrigate today, delay spraying, protect a harvest — at the **Panchayat** level. Rainfall and temperature can vary meaningfully within a single block due to soil type, elevation, and land use, so a block-wide forecast can genuinely mislead a farmer in a specific village.

## Our Solution

AeroAtmos takes a Block-level weather forecast and uses a trained machine learning model to **downscale** it into a Panchayat-level forecast, using each panchayat's:

- Geographic coordinates and elevation
- Soil type
- Land use (paddy, wheat/maize, vegetable, mixed agriculture, fallow)
- Historical average rainfall and temperature

The output is then turned into a **plain-language farmer advisory** (e.g. *"avoid irrigation today"*, *"delay pesticide application"*) — not just a number.

---

## What's in this repo

```
AeroAtmos/
├── backend/
│   ├── data_generator.py     # Generates the synthetic District→Block→Panchayat dataset
│   ├── train_model.py        # Trains the downscaling model and checks it against baselines
│   ├── app.py                 # FastAPI backend serving live predictions
│   ├── requirements.txt       # Pinned versions (Python 3.11+)
│   ├── panchayat_master.csv   # Generated: panchayat hierarchy + geo/soil/land-use data
│   ├── training_data.csv      # Generated: 12-month training examples
│   ├── downscaling_model.joblib  # Generated: trained model
│   └── validation_report.json # Generated: model vs. baseline scores (served at /model/skill)
└── weather-down-live.html     # Interactive dashboard (map, charts, advisory) — talks to the backend
```

## Features

- **Interactive satellite map** — panchayats shown as color-coded markers by rainfall, temperature, humidity, or wind
- **12-month timeline** with play/scrub controls to see seasonal variation
- **Block forecast → AI panchayat forecast comparison** — the core value proposition, shown explicitly
- **Trend and comparison charts** — historical average vs. block forecast vs. AI prediction, and a ranked view of all panchayats in a block
- **Automated farmer advisory** generated from the predicted conditions, using IMD's 24-hour rainfall categories (very light 0.1–2.4 mm, light 2.5–15.5, moderate 15.6–64.4, heavy 64.5–115.5 …)
- **Alerts panel** flagging panchayats with moderate-or-heavier rain or heat
- **Chance of rain** from the model itself: the share of the random forest's trees that predict at least 2.5 mm
- **Model check** shown on the page: error on unseen panchayats vs. simply using the block forecast
- **CSV export** of a block's full 12-month forecast
- **Live FastAPI backend** with interactive Swagger docs at `/docs`

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML Model | scikit-learn (RandomForest on the block-forecast correction + linear lapse-rate correction) |
| Backend API | FastAPI + Uvicorn |
| Data | pandas, numpy |
| Frontend | HTML/CSS/JS, Leaflet.js (map), Chart.js (charts) |

---

## Getting Started

Requires **Python 3.11+**.

### 1. Set up the backend

```bash
cd backend
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

### 2. Generate data and train the model

```bash
python data_generator.py
python train_model.py
```

### 3. Start the API

```bash
uvicorn app:app --reload
```

The API is now live at `http://127.0.0.1:8000` — see interactive docs at `http://127.0.0.1:8000/docs`.
(From the repo root instead: `uvicorn app:app --reload --app-dir backend`. The backend always reads its files from `backend/`.)

### 4. Open the dashboard

Open `weather-down-live.html` directly in your browser (double-click it) while the server from step 3 is running. It connects automatically to `http://127.0.0.1:8000`.
If the backend runs somewhere else, add it to the URL: `weather-down-live.html?api=https://your-server.example.com`.

---

## API Endpoints

| Endpoint | Description |
|---|---|
| `GET /districts` | List all districts |
| `GET /blocks/{district}` | List blocks in a district |
| `GET /panchayats/{block}` | List panchayats in a block |
| `GET /forecast/{panchayat_id}?month=` | Downscaled forecast + advisory for one panchayat (`month` 1–12, defaults to the current month) |
| `GET /forecast/block/{block}?month=` | Downscaled forecast for every panchayat in a block |
| `GET /model/skill` | Validation report: model vs. baseline error on unseen panchayats |

---

## How the model works and how well it does

- **Rainfall:** a RandomForest learns the *correction* to the block forecast (panchayat value minus block value) from each panchayat's location, elevation, soil, land use and history. Predicting the correction keeps it anchored to the block forecast.
- **Temperature:** a linear correction on elevation (a lapse rate learned from the data).
- **Humidity and wind:** the synthetic data has no panchayat-level signal for these, so the model uses the training climatology instead of pretending to learn one.

`train_model.py` scores the model with 5-fold cross-validation **grouped by panchayat** — every score is on panchayats the model never saw — against the obvious baseline:

| Variable | Model MAE | Baseline | Baseline MAE |
|---|---|---|---|
| Rainfall | 2.01 mm | block forecast as-is | 2.27 mm (**model 11% better**) |
| Temperature | 0.42 °C | block forecast as-is | 0.43 °C |
| Humidity | 3.82 % | monthly climatology | 3.82 % |
| Wind | 2.37 km/h | mean wind | 2.37 km/h |

These numbers are on synthetic data, so they show the pipeline works — not real-world accuracy.

---

## Current Scope & Next Steps

This is a working prototype built to prove the downscaling pipeline end-to-end:

- **Data:** Currently synthetic (2 districts, 5 blocks, ~21 panchayats in Bihar) — generated with realistic seasonal and geographic patterns. Real deployment would use IMD's block forecast API and Census/SDMA panchayat boundaries.
- **Model:** scikit-learn only, for zero extra native dependencies. The block-forecast *correction* setup carries over to gradient boosting or spatial models (CNN/GNN) once real data is in.
- **Delivery:** Dashboard is the demo interface; real-world deployment would extend to SMS/IVR/WhatsApp alerts for farmers without smartphone access.
- **Validation:** Currently measured on synthetic data (table above); next step is validation against real ground-station observations (IMD AWS, Bihar Mausam Sewa Kendra rain gauges).

---

## Team AERO VISION

Built for Smart India Hackathon 2026 — Problem Statement 26074.
