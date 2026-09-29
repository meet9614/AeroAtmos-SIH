"""
build_panchayats.py — builds the real panchayat list used by AeroAtmos.

Source: LGD (Local Government Directory) village and block boundaries, as
published by the datameet community at
https://github.com/ramSeraph/indian_admin_boundaries (releases "villages" and
"blocks"; original data from LGD / Bharatmaps). Every village polygon carries
its gram panchayat code and name, so dissolving villages by gram panchayat gives
each panchayat's boundary, centroid and area.

Outputs (committed, so nobody needs to re-run this):
    data/blocks.csv            one row per block: LGD code, name, district, centroid
    data/panchayats.csv        one row per gram panchayat: LGD code, name, block, centroid, area
    data/panchayats.geojson    simplified panchayat boundaries for the map

Run (downloads ~410 MB once, needs `pip install py7zr shapely`):
    python tools/build_panchayats.py
    python tools/build_panchayats.py --blocks MASAURHI BIKRAM ...   # other blocks
"""

import argparse
import io
import json
import math
import subprocess
import shutil
import urllib.request
from collections import defaultdict
from pathlib import Path

import shapely
from shapely.geometry import mapping, shape
from shapely.ops import unary_union

DATA = Path(__file__).resolve().parent.parent / "data"
CACHE = Path(__file__).resolve().parent / ".lgd_cache"
RELEASES = "https://github.com/ramSeraph/indian_admin_boundaries/releases/download"

# Blocks in the demo: (district LGD code, block name as spelled in LGD)
DEFAULT_BLOCKS = [(212, "MASAURHI"), (212, "BIKRAM"), (212, "NAUBATPUR"),
                  (196, "TEKARI"), (196, "SHERGHATTY")]
DISTRICTS = {212: "Patna", 196: "Gaya"}
DISPLAY_NAMES = {"SHERGHATTY": "Sherghati"}  # common spelling for the UI


def download(name: str, tag: str) -> Path:
    CACHE.mkdir(exist_ok=True)
    path = CACHE / name
    if not path.exists():
        print(f"Downloading {name} ...")
        urllib.request.urlretrieve(f"{RELEASES}/{tag}/{name}", path)
    return path


def geojsonl_lines(archive: Path, must_contain: tuple[str, ...]):
    """Stream lines out of a .geojsonl.7z, keeping only lines containing one of must_contain."""
    if shutil.which("7z"):
        proc = subprocess.Popen(["7z", "e", "-so", str(archive)], stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL)
        stream = io.TextIOWrapper(proc.stdout, encoding="utf-8")
    else:
        import py7zr  # slower pure-Python fallback
        with py7zr.SevenZipFile(archive) as z:
            (name, bio), = z.readall().items()
        stream = io.TextIOWrapper(bio, encoding="utf-8")
    for line in stream:
        if any(s in line for s in must_contain):
            yield json.loads(line)


def area_km2(geom) -> float:
    lat = geom.centroid.y
    return geom.area * 111.32 ** 2 * math.cos(math.radians(lat))


def inside_point(geom):
    c = geom.centroid
    return c if geom.contains(c) else geom.representative_point()


def pretty(name: str) -> str:
    return DISPLAY_NAMES.get(name, name.title())


def main(blocks: list[tuple[int, str]]):
    wanted = {(d, b.upper()) for d, b in blocks}
    district_filter = tuple(f'"dist_lgd": {d},' for d in {d for d, _ in wanted})

    # Blocks
    block_rows = []
    for f in geojsonl_lines(download("LGD_Blocks.geojsonl.7z", "blocks"), district_filter):
        p = f["properties"]
        if (p["dist_lgd"], p["block_name"].upper()) in wanted:
            g = shape(f["geometry"])
            pt = inside_point(g)
            block_rows.append({"block_lgd": p["block_lgd"], "block": pretty(p["block_name"]),
                               "district": DISTRICTS.get(p["dist_lgd"], p["district"]),
                               "lat": round(pt.y, 5), "lon": round(pt.x, 5),
                               "area_km2": round(area_km2(g), 1)})

    # Villages -> gram panchayats
    villages = defaultdict(list)
    meta = {}
    for f in geojsonl_lines(download("LGD_Villages.geojsonl.7z", "villages"), district_filter):
        p = f["properties"]
        if (p["dist_lgd"], (p["block_name"] or "").upper()) in wanted and p.get("gp_code"):
            villages[p["gp_code"]].append(shape(f["geometry"]).buffer(0))
            meta[p["gp_code"]] = (p["gp_name"], p["block_lgd"])

    block_by_lgd = {b["block_lgd"]: b for b in block_rows}
    gp_rows, features = [], []
    for code, geoms in sorted(villages.items()):
        name, block_lgd = meta[code]
        b = block_by_lgd[block_lgd]
        g = unary_union(geoms)
        pt = inside_point(g)
        gp_id = f"GP{code}"
        gp_rows.append({"panchayat_id": gp_id, "gp_lgd": code, "panchayat": name.title(),
                        "block": b["block"], "block_lgd": block_lgd, "district": b["district"],
                        "lat": round(pt.y, 5), "lon": round(pt.x, 5),
                        "area_km2": round(area_km2(g), 2), "n_villages": len(geoms)})
        features.append({"type": "Feature",
                         "properties": {"panchayat_id": gp_id, "panchayat": name.title(),
                                        "block": b["block"]},
                         "geometry": mapping(shapely.set_precision(
                             g.simplify(0.0003, preserve_topology=True), grid_size=1e-5))})

    for b in block_rows:
        b["n_panchayats"] = sum(r["block_lgd"] == b["block_lgd"] for r in gp_rows)

    DATA.mkdir(exist_ok=True)
    write_csv(DATA / "blocks.csv", sorted(block_rows, key=lambda r: (r["district"], r["block"])))
    write_csv(DATA / "panchayats.csv", sorted(gp_rows, key=lambda r: (r["district"], r["block"], r["panchayat"])))
    geo = {"type": "FeatureCollection", "features": features}
    (DATA / "panchayats.geojson").write_text(json.dumps(geo, separators=(",", ":")))
    print(f"{len(block_rows)} blocks, {len(gp_rows)} gram panchayats written to {DATA}")


def write_csv(path: Path, rows: list[dict]):
    import csv
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--blocks", nargs="*", help="LGD block names (Patna/Gaya); default: the 5 demo blocks")
    args = ap.parse_args()
    if args.blocks:
        blocks = [(d, b) for b in args.blocks for d in DISTRICTS]
    else:
        blocks = DEFAULT_BLOCKS
    main(blocks)
