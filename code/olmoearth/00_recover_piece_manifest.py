"""
Recover piece job manifest entries lost by a manifest-overwrite bug in
04b_submit_large_states.py.

The piece jobs (state_XX_YYYY_p01, _p02, ...) were submitted to OlmoEarth but
their manifest rows were overwritten.  This script queries the API, finds all
piece jobs in the project, and adds any missing rows to the manifest.

Run once to repair the manifest, then continue with 02_collect_results.py.

Usage:
    uv run python 00_recover_piece_manifest.py
"""
import csv
import json
import os
import re
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

HERE = Path(__file__).parent
load_dotenv(HERE / ".env")
MANIFEST = HERE / "manifest.csv"
CONFIG = json.loads((HERE / "config.json").read_text())

BASE_URL = "https://olmoearth.allenai.org"
HEADERS = {
    "Authorization": f"Bearer {os.environ['OLMOEARTH_API_KEY']}",
    "Content-Type": "application/json",
}

PIECE_RE = re.compile(r"^state_\d{2}_\d{4}_p\d+$")

# Load current manifest
with open(MANIFEST) as fh:
    rows = list(csv.DictReader(fh))
existing_names = {r["name"] for r in rows}
print(f"Manifest has {len(rows)} entries. Looking for missing piece jobs...\n")

recovered: dict[str, dict] = {}


# ── Strategy 1: search completed results ────────────────────────────────────
print("Querying completed results (search API)...")
page = 1
total_results = 0
while True:
    resp = requests.post(
        f"{BASE_URL}/api/v1/prediction-results/search",
        json={"project_id": CONFIG["project_id"], "page": page, "limit": 100},
        headers=HEADERS,
    )
    resp.raise_for_status()
    data = resp.json()
    records = data.get("records", [])
    total_results += len(records)
    for r in records:
        name = r.get("prediction_name", "")
        if PIECE_RE.match(name) and name not in existing_names:
            pid = r.get("prediction_id") or r.get("id", "")
            recovered[name] = {"name": name, "prediction_id": pid, "status": "submitted"}
    if len(records) < 100:
        break
    page += 1
    time.sleep(0.3)

print(f"  Scanned {total_results} completed results, found {len(recovered)} missing piece jobs.")

# Print a sample result to show available fields (helpful for debugging)
if total_results > 0:
    resp2 = requests.post(
        f"{BASE_URL}/api/v1/prediction-results/search",
        json={"project_id": CONFIG["project_id"], "page": 1, "limit": 1},
        headers=HEADERS,
    )
    sample = resp2.json().get("records", [{}])[0]
    print(f"  Sample result fields: {list(sample.keys())}")


# ── Strategy 2: list all predictions to catch in-progress piece jobs ─────────
print("\nQuerying all predictions (listing API)...")
page = 1
in_progress_found = 0
while True:
    resp = requests.get(
        f"{BASE_URL}/api/v1/predictions",
        headers=HEADERS,
        params={"project_id": CONFIG["project_id"], "page": page, "limit": 100},
    )
    if resp.status_code in (404, 405, 422):
        print(f"  Listing endpoint returned {resp.status_code} — not available, skipping.")
        break
    resp.raise_for_status()
    data = resp.json()
    records = data.get("records", [])
    if not records:
        break
    for r in records:
        name = r.get("name", "")
        if PIECE_RE.match(name) and name not in existing_names and name not in recovered:
            recovered[name] = {
                "name": name,
                "prediction_id": r.get("id", ""),
                "status": "submitted",
            }
            in_progress_found += 1
    if len(records) < 100:
        break
    page += 1
    time.sleep(0.3)

if in_progress_found:
    print(f"  Found {in_progress_found} additional in-progress piece jobs.")

# ── Write recovered entries to manifest ──────────────────────────────────────
if recovered:
    rows.extend(recovered.values())
    rows.sort(key=lambda r: r["name"])
    with open(MANIFEST, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["name", "prediction_id", "status"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nAdded {len(recovered)} piece job entries to manifest.")
    print("Run 02_collect_results.py to download them.")
else:
    print("\nNo missing piece jobs found via the API.")
    print("If piece jobs are still processing they may not appear yet —")
    print("re-run this script after a few hours to catch them.")
