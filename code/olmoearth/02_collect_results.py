"""
Single-pass sweep: checks all submitted jobs, downloads and writes COGs for completed ones.
Re-run until manifest shows all done.

For state-level jobs the API may return multiple tile TIFs in the ZIP; these are mosaicked
into a single output COG per job.

Usage:
    uv run python 02_collect_results.py
    # loop until done:
    until uv run python 02_collect_results.py | grep -q "All done"; do sleep 300; done
"""
import csv
import os
import re
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
import rasterio
import requests
from rasterio.merge import merge as rasterio_merge
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv

HERE = Path(__file__).parent
load_dotenv(HERE / ".env")

PROJECT_ROOT = HERE.parent.parent
OUT_DIR = PROJECT_ROOT / "data" / "olmoearth"
MANIFEST = HERE / "manifest.csv"

BASE_URL = "https://olmoearth.allenai.org"
HEADERS = {"Authorization": f"Bearer {os.environ['OLMOEARTH_API_KEY']}"}
POLL_DELAY = 0.5

OUT_DIR.mkdir(parents=True, exist_ok=True)

session = requests.Session()
retry = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
session.mount("https://", HTTPAdapter(max_retries=retry))

with open(MANIFEST) as fh:
    rows = list(csv.DictReader(fh))

pending = [r for r in rows if r["status"] == "submitted"]
done_count = sum(1 for r in rows if r["status"] == "done")
print(f"{done_count} done, {len(pending)} pending")

if not pending:
    print("All done.")
    raise SystemExit(0)


def save_manifest():
    with open(MANIFEST, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["name", "prediction_id", "status"])
        writer.writeheader()
        writer.writerows(rows)


newly_done = 0
for i, row in enumerate(pending, 1):
    name, prediction_id = row["name"], row["prediction_id"]
    print(f"\n[{i}/{len(pending)} pending | {done_count + newly_done} done] {name}")

    resp = session.get(
        f"{BASE_URL}/api/v1/predictions/{prediction_id}",
        headers=HEADERS,
        params={"include_result_metadata": "true"},
    )
    resp.raise_for_status()
    pred = resp.json()["records"][0]
    time.sleep(POLL_DELAY)

    if pred["status"] in ("failed", "cancelled", "error"):
        print(f"  status: {pred['status']}")
        row["status"] = pred["status"]
        save_manifest()
        continue

    if pred["status"] != "completed":
        print(f"  status: {pred['status']} — skipping until next pass")
        continue

    token = pred["result"]["download_token"]

    with tempfile.TemporaryDirectory(dir=OUT_DIR) as tmp:
        zip_path = Path(tmp) / "result.zip"

        # Write auth to a temp file so the API key never appears in ps output.
        cfg_path = Path(tmp) / "curl.cfg"
        cfg_path.write_text(
            f'header = "Authorization: Bearer {os.environ["OLMOEARTH_API_KEY"]}"\n'
        )

        url = (f"{BASE_URL}/api/v1/prediction-results/files"
               f"?download_token={token}")

        # curl's built-in --retry only covers a small set of error codes.
        # Exit 18 (CURLE_PARTIAL_FILE, server closed connection early) and
        # exit 92 (CURLE_HTTP2_STREAM) are not retried by curl itself, so
        # we wrap the call in a Python loop that handles those explicitly.
        RETRIABLE = {18, 56, 92}
        for attempt in range(20):
            result = subprocess.run(
                [
                    "curl",
                    "--http1.1",     # avoid HTTP/2 stream errors
                    "-L",            # follow redirects
                    "--fail",        # non-zero exit on HTTP errors
                    "--config", str(cfg_path),
                    "-o", str(zip_path),
                    url,
                ],
                check=False,
            )
            if result.returncode == 0:
                break
            if result.returncode not in RETRIABLE or attempt == 19:
                raise subprocess.CalledProcessError(result.returncode, "curl")
            print(f"  curl exit {result.returncode} — retry {attempt + 1}/20 in 10s...")
            time.sleep(10)

        with zipfile.ZipFile(zip_path) as zf:
            tif_names = sorted(n for n in zf.namelist() if n.endswith(".tif"))
            zf.extractall(tmp)
        print(f"  [{name}] {len(tif_names)} tile(s) in ZIP")

        tif_paths = [Path(tmp) / n for n in tif_names]

        if len(tif_paths) == 1:
            with rasterio.open(tif_paths[0]) as src:
                profile = src.profile.copy()
                data = src.read()
        else:
            datasets = [rasterio.open(p) for p in tif_paths]
            data, transform = rasterio_merge(datasets)
            profile = datasets[0].profile.copy()
            profile.update(height=data.shape[1], width=data.shape[2], transform=transform)
            for ds in datasets:
                ds.close()

    out_path = OUT_DIR / f"{name}.tif"
    profile.update(
        driver="GTiff",
        compress="deflate",
        predictor=2,
        tiled=True,
        blockxsize=256,
        blockysize=256,
        bigtiff="IF_SAFER",
    )
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(data)

    print(f"  [{name}] → {out_path.name}  shape={data.shape}")
    row["status"] = "done"
    save_manifest()
    newly_done += 1

remaining = sum(1 for r in rows if r["status"] == "submitted")
print(f"\nThis pass: +{newly_done} collected. Remaining: {remaining}")
if remaining > 0:
    print("Re-run to collect more.")

# ── Mosaic completed piece sets into full-state COGs ─────────────────────────
PIECE_RE = re.compile(r"^(state_\d{2}_\d{4})_p\d+$")

piece_groups: dict[str, list] = {}
for r in rows:
    m = PIECE_RE.match(r["name"])
    if m:
        piece_groups.setdefault(m.group(1), []).append(r)

for base_name, piece_rows in sorted(piece_groups.items()):
    state_cog = OUT_DIR / f"{base_name}.tif"
    if state_cog.exists():
        continue
    done = [r for r in piece_rows if r["status"] == "done"]
    if len(done) < len(piece_rows):
        print(f"  {base_name}: {len(done)}/{len(piece_rows)} pieces done — mosaic pending")
        continue

    print(f"\nMosaicking {len(piece_rows)} piece(s) → {base_name}.tif")
    piece_tifs = [OUT_DIR / f"{r['name']}.tif" for r in piece_rows]
    missing_tifs = [p for p in piece_tifs if not p.exists()]
    if missing_tifs:
        print(f"  WARNING: piece TIFs missing: {[p.name for p in missing_tifs]}")
        continue

    datasets = [rasterio.open(p) for p in piece_tifs]
    data, transform = rasterio_merge(datasets)
    profile = datasets[0].profile.copy()
    profile.update(height=data.shape[1], width=data.shape[2], transform=transform)
    for ds in datasets:
        ds.close()

    profile.update(
        driver="GTiff", compress="deflate", predictor=2,
        tiled=True, blockxsize=256, blockysize=256, bigtiff="IF_SAFER",
    )
    with rasterio.open(state_cog, "w", **profile) as dst:
        dst.write(data)
    print(f"  → {state_cog.name}  shape={data.shape}")

pieces_pending = any(
    r["status"] != "done"
    for r in rows
    if PIECE_RE.match(r["name"])
)
if remaining == 0 and not pieces_pending:
    print("All done.")
