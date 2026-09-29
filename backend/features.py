"""
features.py — model inputs for AeroAtmos, shared by train_model.py and app.py.

One row per (panchayat, date, lead day) with:
- bf_*   the block forecast for the panchayat's own block at that lead day
- idw_*  block forecasts interpolated to the panchayat (inverse-distance weights
         over all blocks), which carries the gradient between neighbouring blocks
- the panchayat's location relative to its block centre, size, terrain and
  (if fetched) land cover
- lead day and season
"""

from pathlib import Path

import numpy as np
import pandas as pd

from weather import DAILY

INTERPOLATED = ["rain", "tmax", "tmin", "rh", "wind"]
BF = [f"bf_{v}" for v in DAILY]
IDW = [f"idw_{v}" for v in INTERPOLATED]
PLACE = ["lat", "lon", "dnorth_km", "deast_km", "dist_km", "area_km2"]
OPTIONAL = ["elevation_m", "relief_m", "elev_diff_m",
            "lc_tree", "lc_cropland", "lc_builtup", "lc_water", "lc_other"]
CALENDAR = ["lead_day", "doy_sin", "doy_cos"]


def load_places(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    gps = pd.read_csv(data_dir / "panchayats.csv")
    blocks = pd.read_csv(data_dir / "blocks.csv")
    static = data_dir / "static_features.csv"
    if static.exists():
        gps = gps.merge(pd.read_csv(static), on="panchayat_id", how="left")
    centre = blocks.set_index("block_lgd")[["lat", "lon"]]
    blat, blon = gps.block_lgd.map(centre.lat), gps.block_lgd.map(centre.lon)
    gps["dnorth_km"] = (gps.lat - blat) * 111.0
    gps["deast_km"] = (gps.lon - blon) * 111.0 * np.cos(np.radians(gps.lat))
    gps["dist_km"] = np.hypot(gps.dnorth_km, gps.deast_km)
    return gps, blocks


def feature_columns(gps: pd.DataFrame) -> list[str]:
    return BF + IDW + PLACE + [c for c in OPTIONAL if c in gps] + CALENDAR


def idw_weights(gps: pd.DataFrame, blocks: pd.DataFrame, power: float = 2.0) -> np.ndarray:
    dn = (gps.lat.to_numpy()[:, None] - blocks.lat.to_numpy()[None, :]) * 111.0
    de = (gps.lon.to_numpy()[:, None] - blocks.lon.to_numpy()[None, :]) * 111.0 * np.cos(np.radians(25.0))
    dist = np.maximum(np.hypot(dn, de), 1.0)  # km; floor avoids division by ~0
    return 1.0 / dist ** power                # shape (panchayats, blocks)


def build(block_fc: pd.DataFrame, gps: pd.DataFrame, blocks: pd.DataFrame) -> pd.DataFrame:
    """block_fc: rows (block_lgd, date, lead_day, rain, tmax, ...). Returns one row per panchayat/date/lead."""
    block_fc = block_fc.copy()
    block_fc["date"] = pd.to_datetime(block_fc["date"])
    own = block_fc.rename(columns={v: f"bf_{v}" for v in DAILY if v in block_fc})
    rows = gps.merge(own, on="block_lgd", how="inner")
    for c in BF:  # a variable the forecast source did not provide stays as missing values
        if c not in rows:
            rows[c] = np.nan

    # inverse-distance interpolation of each variable across blocks
    order = blocks.block_lgd.tolist()
    w = idw_weights(gps, blocks)
    keys = block_fc[["date", "lead_day"]].drop_duplicates().sort_values(["date", "lead_day"])
    idx = pd.MultiIndex.from_frame(keys)
    interp = []
    for v in INTERPOLATED:
        if v not in block_fc:
            interp.append(np.full((len(idx), len(gps)), np.nan))
            continue
        wide = block_fc.pivot_table(index=["date", "lead_day"], columns="block_lgd", values=v,
                                    aggfunc="first").reindex(index=idx, columns=order)
        vals = wide.to_numpy()                              # (time, blocks)
        mask = ~np.isnan(vals)
        num = np.nan_to_num(vals) @ w.T                     # (time, panchayats)
        den = mask.astype(float) @ w.T
        interp.append(num / np.where(den > 0, den, np.nan))
    n_t, n_g = len(idx), len(gps)
    long = pd.DataFrame({
        "panchayat_id": np.tile(gps.panchayat_id.to_numpy(), n_t),
        "date": np.repeat(keys.date.to_numpy(), n_g),
        "lead_day": np.repeat(keys.lead_day.to_numpy(), n_g),
        **{f"idw_{v}": arr.reshape(-1) for v, arr in zip(INTERPOLATED, interp)},
    })
    rows = rows.merge(long, on=["panchayat_id", "date", "lead_day"], how="left")

    doy = rows.date.dt.dayofyear.to_numpy()
    rows["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    rows["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return rows
