"""
train_model.py — trains AeroAtmos's panchayat downscaling models on the data
from fetch_real_data.py, and checks them against simple baselines.

Model
-----
For each variable IMD's agromet advisories use (rainfall, max/min temperature,
humidity, wind), a gradient-boosted model learns the *correction* to the block
forecast for a panchayat: (panchayat value - block forecast), from the block
forecast itself, block forecasts interpolated from neighbouring blocks, the
panchayat's position, size and terrain, the lead day (1-5) and the season.
A classifier gives the chance of at least 2.5 mm of rain (IMD "light rain").

Validation (honest by construction)
-----------------------------------
- Time: the last 180 days are held out; models only see earlier days.
- Space: a quarter of the panchayats in every block are never used in
  training, so we can score places the model has never seen.
Baselines on exactly the same rows:
- block forecast as-is (what a farmer gets today)
- bias-corrected block forecast (average correction per lead day and month)
- distance-weighted interpolation between block forecasts
Metrics: MAE per variable and lead day; for rain also hit rate / false alarms
at IMD thresholds, and the Brier score of the chance of rain.

Run:  python train_model.py
Outputs: models/downscaling.joblib, validation_report.json
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

import features as F
from weather import TARGETS

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
HOLDOUT_DAYS = 180
RAIN_DAY_MM = 2.5
RAIN_THRESHOLDS = {"light rain or more (>=2.5 mm)": 2.5, "moderate rain or more (>=15.6 mm)": 15.6}
LIMITS = {"rain": (0, None), "rh": (0, 100), "wind": (0, None)}
NAMES = {"rain": "Rainfall (mm)", "tmax": "Max temperature (°C)", "tmin": "Min temperature (°C)",
         "rh": "Humidity (%)", "wind": "Wind (km/h)"}
BOOSTING = dict(learning_rate=0.05, max_iter=500, max_leaf_nodes=31, min_samples_leaf=40,
                l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
                n_iter_no_change=30, random_state=0)


def clip(v: str, x):
    lo, hi = LIMITS.get(v, (None, None))
    return np.clip(x, lo, hi) if (lo is not None or hi is not None) else x


def load_table():
    gps, blocks = F.load_places(DATA)
    bf = pd.read_csv(DATA / "block_forecasts.csv")
    obs = pd.read_csv(DATA / "panchayat_obs.csv", parse_dates=["date"])
    table = F.build(bf, gps, blocks).merge(obs.rename(columns={v: f"obs_{v}" for v in TARGETS}),
                                           on=["panchayat_id", "date"], how="inner")
    return table, F.feature_columns(gps), gps


def split(table: pd.DataFrame, gps: pd.DataFrame):
    cutoff = table.date.max() - pd.Timedelta(days=HOLDOUT_DAYS)
    rng = np.random.default_rng(42)
    unseen = set()
    for _, grp in gps.groupby("block_lgd"):
        ids = grp.panchayat_id.to_numpy()
        unseen |= set(rng.choice(ids, size=max(1, round(len(ids) / 4)), replace=False).tolist())
    is_unseen = table.panchayat_id.isin(unseen)
    train = table[(table.date <= cutoff) & ~is_unseen]
    test = table[table.date > cutoff]
    return train, test, test[test.panchayat_id.isin(unseen)], cutoff, sorted(unseen)


def mae(a, b):
    m = ~(np.isnan(a) | np.isnan(b))
    return float(np.mean(np.abs(a[m] - b[m])))


def categorical(obs, pred, thr):
    o, p = obs >= thr, pred >= thr
    hits, misses, false = int((o & p).sum()), int((o & ~p).sum()), int((~o & p).sum())
    return {"hit_rate": round(hits / max(hits + misses, 1), 3),
            "false_alarm_ratio": round(false / max(hits + false, 1), 3),
            "critical_success_index": round(hits / max(hits + misses + false, 1), 3),
            "observed_events": hits + misses}


def evaluate(train, test_sets, feats):
    report = {"variables": {}, "rain_categories": {}, "rain_chance": {}}
    for v in TARGETS:
        tr = train.dropna(subset=[f"obs_{v}", f"bf_{v}"])
        y = tr[f"obs_{v}"] - tr[f"bf_{v}"]
        model = HistGradientBoostingRegressor(**BOOSTING).fit(tr[feats], y)
        bias = y.groupby([tr.lead_day, tr.date.dt.month]).mean()
        report["variables"][v] = {"name": NAMES[v]}
        for label, te in test_sets.items():
            te = te.dropna(subset=[f"obs_{v}", f"bf_{v}"])
            obs, raw = te[f"obs_{v}"].to_numpy(), te[f"bf_{v}"].to_numpy()
            keys = pd.MultiIndex.from_arrays([te.lead_day, te.date.dt.month])
            corr = bias.reindex(keys).fillna(float(y.mean())).to_numpy()
            preds = {
                "model": clip(v, raw + model.predict(te[feats])),
                "block_forecast": raw,
                "bias_corrected_block": clip(v, raw + corr),
                "interpolated_blocks": te[f"idw_{v}"].to_numpy(),
            }
            scores = {k: round(mae(obs, p), 3) for k, p in preds.items()}
            leads = te.lead_day.to_numpy()
            by_lead = {int(d): {"model": round(mae(obs[leads == d], preds["model"][leads == d]), 3),
                                "block_forecast": round(mae(obs[leads == d], raw[leads == d]), 3)}
                       for d in sorted(set(leads))}
            best_base = min(s for k, s in scores.items() if k != "model")
            report["variables"][v][label] = {
                "mae": scores, "rows": int(len(te)), "mae_by_lead_day": by_lead,
                "improvement_vs_block_forecast_pct": round(100 * (1 - scores["model"] / scores["block_forecast"]), 1),
                "improvement_vs_best_baseline_pct": round(100 * (1 - scores["model"] / best_base), 1)}
            if v == "rain":
                report["rain_categories"][label] = {
                    name: {"model": categorical(obs, preds["model"], t),
                           "block_forecast": categorical(obs, raw, t)}
                    for name, t in RAIN_THRESHOLDS.items()}

    # chance of rain (>= 2.5 mm)
    tr = train.dropna(subset=["obs_rain"])
    clf = HistGradientBoostingClassifier(**BOOSTING).fit(tr[feats], tr.obs_rain >= RAIN_DAY_MM)
    clim = (tr.obs_rain >= RAIN_DAY_MM).groupby(tr.date.dt.month).mean()
    for label, te in test_sets.items():
        te = te.dropna(subset=["obs_rain"])
        y = (te.obs_rain >= RAIN_DAY_MM).to_numpy(float)
        probs = {"model": clf.predict_proba(te[feats])[:, 1],
                 "block_forecast_yes_no": (te.bf_rain >= RAIN_DAY_MM).to_numpy(float),
                 "monthly_climatology": te.date.dt.month.map(clim).to_numpy(float)}
        report["rain_chance"][label] = {
            "brier": {k: round(float(np.mean((p - y) ** 2)), 4) for k, p in probs.items()},
            "observed_rain_day_rate": round(float(y.mean()), 3)}
    return report


def fit_final(table, feats):
    models = {}
    for v in TARGETS:
        t = table.dropna(subset=[f"obs_{v}", f"bf_{v}"])
        models[v] = HistGradientBoostingRegressor(**BOOSTING).fit(t[feats], t[f"obs_{v}"] - t[f"bf_{v}"])
    t = table.dropna(subset=["obs_rain"])
    return {"models": models,
            "rain_chance": HistGradientBoostingClassifier(**BOOSTING).fit(t[feats], t.obs_rain >= RAIN_DAY_MM),
            "features": feats, "limits": LIMITS}


def print_summary(report):
    s = report["split"]
    print(f"\nTest period {s['test_period']}, on {s['unseen_panchayats']} of {s['panchayats']} "
          f"panchayats never seen in training.  Mean absolute error:")
    print(f"  {'variable':22s} {'model':>7s} {'block fc':>9s} {'bias-corr':>10s} {'interp':>8s}   model vs block")
    for v, r in report["variables"].items():
        m = r["unseen_panchayats"]["mae"]
        print(f"  {r['name']:22s} {m['model']:7.2f} {m['block_forecast']:9.2f} "
              f"{m['bias_corrected_block']:10.2f} {m['interpolated_blocks']:8.2f}   "
              f"{r['unseen_panchayats']['improvement_vs_block_forecast_pct']:+.1f}%")
    b = report["rain_chance"]["unseen_panchayats"]["brier"]
    print(f"  Chance of rain (Brier, lower is better): model {b['model']:.3f}, "
          f"block yes/no {b['block_forecast_yes_no']:.3f}, climatology {b['monthly_climatology']:.3f}")


if __name__ == "__main__":
    table, feats, gps = load_table()
    train, test_all, test_unseen, cutoff, unseen = split(table, gps)
    meta_file = DATA / "fetch_meta.json"
    meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}
    print(f"{len(table):,} panchayat-day-lead rows; training on {len(train):,}")

    report = evaluate(train, {"unseen_panchayats": test_unseen, "all_panchayats": test_all}, feats)
    report["split"] = {
        "train_period": f"{table.date.min():%Y-%m-%d} to {cutoff:%Y-%m-%d}",
        "test_period": f"{cutoff + pd.Timedelta(days=1):%Y-%m-%d} to {table.date.max():%Y-%m-%d}",
        "panchayats": int(gps.panchayat_id.nunique()), "unseen_panchayats": len(unseen),
        "unseen_panchayat_ids": unseen}
    report["data"] = {
        "source": meta.get("source", "unknown"),
        "block_forecast": f"{meta.get('forecast_model', '?')} via Open-Meteo (stand-in for IMD block forecast)",
        "truth": f"{meta.get('truth_model', '?')} archive via Open-Meteo",
        "fetched_at": meta.get("fetched_at")}
    (BASE / "validation_report.json").write_text(json.dumps(report, indent=2))
    print_summary(report)

    (BASE / "models").mkdir(exist_ok=True)
    joblib.dump(fit_final(table, feats), BASE / "models" / "downscaling.joblib", compress=3)
    print("\nSaved models/downscaling.joblib and validation_report.json")
