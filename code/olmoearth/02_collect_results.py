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
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
import rasterio
import requests
from rasterio.merge import merge as rasterio_merge
from requests.adapters import HTTPAdapter
from tqdm import tqdm
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


# Episode-based retry: up to MAX_EPISODE_ATTEMPTS per uninterrupted failure burst.
# If more than EPISODE_WINDOW seconds pass between failures the counter resets,
# so a large-state download that drops multiple times hours apart is not penalised.
MAX_EPISODE_ATTEMPTS = 5
EPISODE_WINDOW = 60  # seconds of quiet before treating the next failure as a new episode

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

    with tempfile.TemporaryDirectory() as tmp:
        zip_path = Path(tmp) / "result.zip"
        episode_attempts = 0
        last_failure_time = 0.0
        while True:
            if episode_attempts > 0 and (time.time() - last_failure_time) > EPISODE_WINDOW:
                episode_attempts = 0
            try:
                resp = session.get(
                    f"{BASE_URL}/api/v1/prediction-results/files",
                    headers=HEADERS,
                    params={"download_token": token},
                    stream=True,
                )
                resp.raise_for_status()
                total_bytes = int(resp.headers.get("content-length", 0)) or None
                with tqdm(total=total_bytes, unit="B", unit_scale=True,
                          desc="  downloading", leave=False) as pbar:
                    with open(zip_path, "wb") as fh:
                        for chunk in resp.iter_content(chunk_size=65536):
                            fh.write(chunk)
                            pbar.update(len(chunk))
                break
            except (requests.exceptions.ChunkedEncodingError,
                    requests.exceptions.ConnectionError) as e:
                episode_attempts += 1
                last_failure_time = time.time()
                if episode_attempts >= MAX_EPISODE_ATTEMPTS:
                    raise
                wait = 2 ** (episode_attempts - 1)
                print(f"  download interrupted — retrying in {wait}s "
                      f"(episode attempt {episode_attempts}/{MAX_EPISODE_ATTEMPTS})")
                time.sleep(wait)

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
if remaining == 0:
    print("All done.")
else:
    print("Re-run to collect more.")
