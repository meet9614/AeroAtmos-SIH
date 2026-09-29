"""
fetch_real_data.py — downloads the real data AeroAtmos trains on.
Run it on a computer with internet (it takes ~10-20 minutes the first time).

Everything comes from Open-Meteo (https://open-meteo.com): free for
non-commercial use, no API key.

1. Elevation (Copernicus 90 m DEM) at every gram panchayat centroid, plus four
   points ~2 km around it (terrain relief), and at every block centroid.
2. Block forecasts: what NOAA GFS forecast 1-5 days ahead at each block centroid,
   for every day since Jan 2024 (Open-Meteo "Previous Runs" archive). This is the
   stand-in for IMD's block-level forecast feed.
3. Panchayat weather "truth": the ECMWF IFS 9 km analysis archive at each
   panchayat's grid cell (--truth-model era5_land is the alternative). Replace
   with station data (IMD AWS, Bihar Mausam Sewa Kendra rain gauges) when
   available - a ~9 km grid is still coarser than a panchayat.
4. The current 5-day forecast for each block (the API falls back to this when it
   cannot reach the internet during a demo).
5. Optional: land cover shares per panchayat from ESA WorldCover 10 m
   (needs `pip install rasterio`; skipped otherwise).

Usage:
    python fetch_real_data.py --check      # 10-second connectivity test
    python fetch_real_data.py              # full download; re-runs reuse data/cache/
Outputs (in data/): static_features.csv, block_forecasts.csv, panchayat_obs.csv,
                    live_block_forecast.json, fetch_meta.json
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import weather as W

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
CACHE = DATA / "cache"
START = "2024-01-01"  # Previous Runs archive starts in Jan 2024 for most models
PAUSE = 3.0  # seconds between big requests: keeps us under Open-Meteo's free per-minute limit
WORLDCOVER = ("https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
              "ESA_WorldCover_10m_2021_v200_{tile}_Map.tif")


def log(msg: str):
    print(msg, flush=True)


def date_chunks(start: str, end: str, days: int):
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    while s <= e:
        c = min(s + pd.Timedelta(days=days - 1), e)
        yield s.strftime("%Y-%m-%d"), c.strftime("%Y-%m-%d")
        s = c + pd.Timedelta(days=1)


def fetch_static(gps: pd.DataFrame, blocks: pd.DataFrame) -> pd.DataFrame:
    offsets = [(0, 0), (0.018, 0), (-0.018, 0), (0, 0.018), (0, -0.018)]  # centre + ~2 km N/S/E/W
    pts = [(g.panchayat_id, g.lat + a, g.lon + b) for _, g in gps.iterrows() for a, b in offsets]
    pts += [(f"B{b.block_lgd}", b.lat, b.lon) for _, b in blocks.iterrows()]
    elev = []
    for i in range(0, len(pts), 100):
        chunk = pts[i:i + 100]
        r = W.get_json(W.ELEVATION_URL, {"latitude": ",".join(f"{p[1]:.5f}" for p in chunk),
                                         "longitude": ",".join(f"{p[2]:.5f}" for p in chunk)}, CACHE)
        elev += r["elevation"]
    pts = pd.DataFrame(pts, columns=["id", "lat", "lon"]).assign(elev=elev)
    block_elev = {int(i[1:]): e for i, e in zip(pts.id, pts.elev) if i.startswith("B")}
    g = pts[~pts.id.str.startswith("B")].groupby("id").elev
    static = pd.DataFrame({"elevation_m": g.first(), "relief_m": g.max() - g.min()})
    static.index.name = "panchayat_id"
    static = static.reset_index().merge(gps[["panchayat_id", "block_lgd"]], on="panchayat_id")
    static["block_elevation_m"] = static.block_lgd.map(block_elev)
    static["elev_diff_m"] = static.elevation_m - static.block_elevation_m
    return static.drop(columns="block_lgd").round(1)


def fetch_landcover(geojson: Path) -> pd.DataFrame | None:
    try:
        import rasterio
        from rasterio.mask import mask
    except ImportError:
        log("  rasterio not installed -> skipping land cover (pip install rasterio to add it)")
        return None
    classes = {"lc_tree": [10], "lc_cropland": [40], "lc_builtup": [50], "lc_water": [80, 90],
               "lc_other": [20, 30, 60, 70, 95, 100]}
    rows, open_tiles = [], {}
    feats = json.loads(geojson.read_text())["features"]
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif"):
        for f in feats:
            lon, lat = _first_coord(f["geometry"])
            tile = f"N{int(math.floor(lat / 3) * 3):02d}E{int(math.floor(lon / 3) * 3):03d}"
            if tile not in open_tiles:
                open_tiles[tile] = rasterio.open(WORLDCOVER.format(tile=tile))
            out, _ = mask(open_tiles[tile], [f["geometry"]], crop=True, nodata=0)
            vals = out[0][out[0] > 0]
            total = max(len(vals), 1)
            rows.append({"panchayat_id": f["properties"]["panchayat_id"],
                         **{k: round(float(np.isin(vals, v).sum()) / total, 3) for k, v in classes.items()}})
    for src in open_tiles.values():
        src.close()
    return pd.DataFrame(rows)


def _first_coord(geom):
    c = geom["coordinates"]
    while isinstance(c[0], list):
        c = c[0]
    return c[0], c[1]


def fetch_block_forecasts(blocks: pd.DataFrame, start: str, end: str, model: str) -> pd.DataFrame:
    frames = []
    for i, (_, b) in enumerate(blocks.iterrows(), 1):
        log(f"  block forecasts {i}/{len(blocks)}: {b.block}")
        for s, e in date_chunks(start, end, 92):
            h = W.hourly_frame(W.get_json(W.PREVIOUS_RUNS_URL,
                                          W.previous_runs_params(b.lat, b.lon, s, e, model), CACHE,
                                          pause=PAUSE))
            for lead, hl in W.split_leads(h).items():
                d = W.daily_from_hourly(hl).reset_index()
                frames.append(d.assign(block_lgd=b.block_lgd, lead_day=lead))
    df = pd.concat(frames, ignore_index=True)
    cols = [c for c in W.DAILY if c in df]
    return df[["block_lgd", "date", "lead_day"] + cols].round({c: 3 for c in cols})


def fetch_panchayat_obs(gps: pd.DataFrame, start: str, end: str, model: str) -> pd.DataFrame:
    # ~9-11 km grids: fetch each 0.1-degree cell once and share it between the panchayats in it
    gps = gps.assign(cell_lat=gps.lat.round(1), cell_lon=gps.lon.round(1))
    cells = gps[["cell_lat", "cell_lon"]].drop_duplicates().reset_index(drop=True)
    frames = []
    for i, c in cells.iterrows():
        log(f"  panchayat weather: grid cell {i + 1}/{len(cells)} ({c.cell_lat:.1f}, {c.cell_lon:.1f})")
        parts = []
        for s, e in date_chunks(start, end, 366):
            resp = W.get_json(W.ARCHIVE_URL, {"latitude": c.cell_lat, "longitude": c.cell_lon,
                                              "start_date": s, "end_date": e,
                                              "hourly": ",".join(W.TRUTH_HOURLY), "models": model,
                                              "timezone": W.TZ}, CACHE, pause=PAUSE)
            parts.append(W.daily_from_hourly(W.hourly_frame(resp)))
        daily = pd.concat(parts).reset_index()
        members = gps[(gps.cell_lat == c.cell_lat) & (gps.cell_lon == c.cell_lon)].panchayat_id
        frames += [daily.assign(panchayat_id=pid) for pid in members]
    df = pd.concat(frames, ignore_index=True)
    cols = [c for c in W.TARGETS if c in df]
    return df[["panchayat_id", "date"] + cols].round({c: 3 for c in cols})


def check():
    b = pd.read_csv(DATA / "blocks.csv").iloc[0]
    end = (pd.Timestamp.now() - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    start = (pd.Timestamp(end) - pd.Timedelta(days=2)).strftime("%Y-%m-%d")
    quick = {"retries": 1, "timeout": 20, "pause": 0}
    tests = {
        "elevation": lambda: W.get_json(W.ELEVATION_URL, {"latitude": b.lat, "longitude": b.lon}, **quick),
        "previous runs (block forecasts)": lambda: W.get_json(
            W.PREVIOUS_RUNS_URL, W.previous_runs_params(b.lat, b.lon, start, end, W.FORECAST_MODEL), **quick),
        "archive (ECMWF IFS 9 km)": lambda: W.get_json(W.ARCHIVE_URL, {
            "latitude": b.lat, "longitude": b.lon, "start_date": start, "end_date": end,
            "hourly": ",".join(W.TRUTH_HOURLY), "models": W.TRUTH_MODEL, "timezone": W.TZ}, **quick),
        "live forecast": lambda: W.fetch_live_block_forecast(pd.DataFrame([b]), **quick),
    }
    ok = True
    for name, fn in tests.items():
        try:
            fn()
            log(f"  OK    {name}")
        except Exception as exc:  # noqa: BLE001
            ok = False
            log(f"  FAIL  {name}: {exc}")
    sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="quick connectivity test only")
    ap.add_argument("--start", default=START)
    ap.add_argument("--end", default=None, help="default: 8 days ago (reanalysis archives lag a few days)")
    ap.add_argument("--forecast-model", default=W.FORECAST_MODEL)
    ap.add_argument("--truth-model", default=W.TRUTH_MODEL)
    ap.add_argument("--no-landcover", action="store_true")
    args = ap.parse_args()
    if args.check:
        check()

    end = args.end or (pd.Timestamp.now(tz=W.TZ) - pd.Timedelta(days=8)).strftime("%Y-%m-%d")
    blocks = pd.read_csv(DATA / "blocks.csv")
    gps = pd.read_csv(DATA / "panchayats.csv")
    t0 = time.time()
    log(f"AeroAtmos data download: {len(gps)} panchayats in {len(blocks)} blocks, {args.start} to {end}"
        + ("   [MOCK MODE - simulated data]" if W.MOCK else ""))

    log("1/5 elevation and terrain")
    static = fetch_static(gps, blocks)
    if not args.no_landcover and not W.MOCK:
        log("2/5 land cover (ESA WorldCover)")
        try:
            lc = fetch_landcover(DATA / "panchayats.geojson")
            if lc is not None:
                static = static.merge(lc, on="panchayat_id", how="left")
        except Exception as exc:  # noqa: BLE001
            log(f"  land cover skipped: {exc}")
    static.to_csv(DATA / "static_features.csv", index=False)

    log(f"3/5 archived block forecasts ({args.forecast_model}, lead 1-5 days)")
    bf = fetch_block_forecasts(blocks, args.start, end, args.forecast_model)
    bf.to_csv(DATA / "block_forecasts.csv", index=False)

    log(f"4/5 panchayat weather ({args.truth_model})")
    obs = fetch_panchayat_obs(gps, args.start, end, args.truth_model)
    obs.to_csv(DATA / "panchayat_obs.csv", index=False)

    log("5/5 current 5-day block forecast")
    live = W.fetch_live_block_forecast(blocks, args.forecast_model)
    (DATA / "live_block_forecast.json").write_text(json.dumps(live, indent=1))

    meta = {"fetched_at": pd.Timestamp.now(tz=W.TZ).strftime("%Y-%m-%d %H:%M IST"),
            "source": "mock" if W.MOCK else "open-meteo", "start": args.start, "end": end,
            "forecast_model": args.forecast_model, "truth_model": args.truth_model,
            "block_forecast_rows": len(bf), "panchayat_obs_rows": len(obs),
            "landcover": any(c.startswith("lc_") for c in static.columns)}
    (DATA / "fetch_meta.json").write_text(json.dumps(meta, indent=2))
    log(f"Done in {time.time() - t0:.0f}s: {len(bf):,} block-forecast rows, {len(obs):,} panchayat-day rows.")
    log("Next: python train_model.py")


if __name__ == "__main__":
    try:
        main()
    except W.OpenMeteoError as exc:
        log(f"\nCould not get data from Open-Meteo: {exc}")
        log("Check your internet connection and run the script again - finished pieces are cached "
            "in data/cache/, so it resumes where it stopped.")
        sys.exit(2)
