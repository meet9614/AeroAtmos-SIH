"""
train_model.py
Trains the AeroAtmos downscaling model and checks it against simple baselines.

Model
-----
- Rainfall: a RandomForest learns the *correction* to the block forecast
  (panchayat value minus block value) from each panchayat's features.
  Predicting the correction keeps the model anchored to the block forecast; it
  gave lower rainfall error than predicting raw values.
- Temperature: a linear correction on elevation (a lapse rate learned from the
  data). The RandomForest could not beat the block forecast for temperature.
- Humidity and wind: in the synthetic data these depend only on the season plus
  random noise, so there is nothing panchayat-specific to learn. The honest
  model is the training climatology (monthly mean humidity, mean wind). Replace
  this with a learned model once real observations are available.

Validation
----------
5-fold cross-validation grouped by panchayat, so every score is measured on
panchayats the model never saw during training (the real downscaling use case).
Baselines:
- rainfall, temperature: use the block forecast unchanged
- humidity: monthly climatology;  wind: overall mean
Results are printed and saved to validation_report.json (served at /model/skill).

Run:
    python data_generator.py   # first, to create training_data.csv
    python train_model.py
Produces: downscaling_model.joblib, validation_report.json
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE = Path(__file__).resolve().parent

NUMERIC = ["month", "block_forecast_rain_mm", "block_forecast_temp_c",
           "lat", "lon", "elevation_m", "hist_avg_rainfall_mm", "hist_avg_temp_c"]
CATEGORICAL = ["soil_type", "land_use"]
FEATURES = NUMERIC + CATEGORICAL


def make_rain_model() -> Pipeline:
    preprocess = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ], remainder="passthrough")
    forest = RandomForestRegressor(n_estimators=300, min_samples_leaf=10,
                                   random_state=42, n_jobs=-1)
    return Pipeline([("prep", preprocess), ("reg", forest)])


def fit(df: pd.DataFrame) -> dict:
    return {
        "rain_correction": make_rain_model().fit(
            df[FEATURES], df.true_rainfall_mm - df.block_forecast_rain_mm),
        "temp_correction": LinearRegression().fit(
            df[["elevation_m"]], df.true_temp_c - df.block_forecast_temp_c),
        "humidity_by_month": df.groupby("month").true_humidity_pct.mean().to_dict(),
        "wind_mean": float(df.true_wind_kmph.mean()),
        "features": FEATURES,
    }


def predict(bundle: dict, X: pd.DataFrame) -> dict:
    rain_corr = bundle["rain_correction"].predict(X[bundle["features"]])
    temp_corr = bundle["temp_correction"].predict(X[["elevation_m"]])
    return {
        "rainfall_mm": np.clip(X.block_forecast_rain_mm.to_numpy() + rain_corr, 0, None),
        "temperature_c": X.block_forecast_temp_c.to_numpy() + temp_corr,
        "humidity_pct": X.month.map(bundle["humidity_by_month"]).to_numpy(),
        "wind_kmph": np.full(len(X), bundle["wind_mean"]),
    }


def validate(df: pd.DataFrame) -> dict:
    truth = {"rainfall_mm": "true_rainfall_mm", "temperature_c": "true_temp_c",
             "humidity_pct": "true_humidity_pct", "wind_kmph": "true_wind_kmph"}
    scores = {k: {"model": [], "baseline": []} for k in truth}
    for train_idx, test_idx in GroupKFold(n_splits=5).split(df, groups=df.panchayat_id):
        tr, te = df.iloc[train_idx], df.iloc[test_idx]
        pred = predict(fit(tr), te)
        baseline = {
            "rainfall_mm": te.block_forecast_rain_mm,
            "temperature_c": te.block_forecast_temp_c,
            "humidity_pct": te.month.map(tr.groupby("month").true_humidity_pct.mean()),
            "wind_kmph": np.full(len(te), tr.true_wind_kmph.mean()),
        }
        for k, col in truth.items():
            scores[k]["model"].append(mean_absolute_error(te[col], pred[k]))
            scores[k]["baseline"].append(mean_absolute_error(te[col], baseline[k]))

    names = {"rainfall_mm": "block forecast", "temperature_c": "block forecast",
             "humidity_pct": "monthly climatology", "wind_kmph": "mean wind"}
    metrics = {}
    for k in truth:
        m, b = float(np.mean(scores[k]["model"])), float(np.mean(scores[k]["baseline"]))
        metrics[k] = {"model_mae": round(m, 2), "baseline": names[k], "baseline_mae": round(b, 2),
                      "improvement_pct": round(100 * (b - m) / b, 1)}
    return {
        "data": "synthetic data",
        "method": "5-fold cross-validation grouped by panchayat (scores on unseen panchayats)",
        "rows": len(df),
        "metrics": metrics,
    }


if __name__ == "__main__":
    df = pd.read_csv(BASE / "training_data.csv")

    report = validate(df)
    print(f"Validation on unseen panchayats ({report['method']}):")
    print(f"  {'variable':15s} {'model MAE':>10s} {'baseline MAE':>13s}  baseline")
    for k, m in report["metrics"].items():
        print(f"  {k:15s} {m['model_mae']:10.2f} {m['baseline_mae']:13.2f}  {m['baseline']}"
              f"  ({m['improvement_pct']:+.1f}%)")
    (BASE / "validation_report.json").write_text(json.dumps(report, indent=2))

    joblib.dump(fit(df), BASE / "downscaling_model.joblib", compress=3)
    print("\nSaved model -> downscaling_model.joblib, report -> validation_report.json")
