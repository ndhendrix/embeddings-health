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
    t0 = time.time()
    resp = requests.post(
        f"{BASE_URL}/api/v1/prediction-results/search",
        json={"project_id": CONFIG["project_id"], "page": page, "limit": 100},
        headers=HEADERS,
    )
    elapsed = time.time() - t0
    print(f"  page {page}: HTTP {resp.status_code} in {elapsed:.1f}s", end="")
    resp.raise_for_status()
    data = resp.json()
    records = data.get("records", [])
    total_results += len(records)
    total_pages = data.get("total_pages") or data.get("pages") or "?"
    new_this_page = 0
    for r in records:
        name = r.get("prediction_name", "")
        if PIECE_RE.match(name) and name not in existing_names:
            pid = r.get("prediction_id") or r.get("id", "")
            recovered[name] = {"name": name, "prediction_id": pid, "status": "submitted"}
            new_this_page += 1
    print(f" — {len(records)} records, {new_this_page} new pieces"
          f" (total: {total_results} scanned, {len(recovered)} pieces found)"
          f" [page {page}/{total_pages}]")
    if page == 1 and records:
        sample = records[0]
        print(f"  Sample result fields: {list(sample.keys())}")
    if len(records) < 100:
        break
    page += 1
    time.sleep(0.3)

print(f"\nSearch API done: {total_results} results scanned, {len(recovered)} piece jobs found.\n")


# ── Strategy 2: list all predictions to catch in-progress piece jobs ─────────
print("Querying all predictions (listing API)...")
page = 1
in_progress_found = 0
while True:
    t0 = time.time()
    resp = requests.get(
        f"{BASE_URL}/api/v1/predictions",
        headers=HEADERS,
        params={"project_id": CONFIG["project_id"], "page": page, "limit": 100},
    )
    elapsed = time.time() - t0
    print(f"  page {page}: HTTP {resp.status_code} in {elapsed:.1f}s", end="")
    if resp.status_code in (404, 405, 422):
        print(f" — endpoint not available, skipping.")
        break
    resp.raise_for_status()
    data = resp.json()
    records = data.get("records", [])
    total_pages = data.get("total_pages") or data.get("pages") or "?"
    new_this_page = 0
    for r in records:
        name = r.get("name", "")
        if PIECE_RE.match(name) and name not in existing_names and name not in recovered:
            recovered[name] = {
                "name": name,
                "prediction_id": r.get("id", ""),
                "status": "submitted",
            }
            in_progress_found += 1
            new_this_page += 1
    print(f" — {len(records)} records, {new_this_page} new pieces [page {page}/{total_pages}]")
    if page == 1 and records:
        sample = records[0]
        print(f"  Sample result fields: {list(sample.keys())}")
    if len(records) < 100:
        break
    page += 1
    time.sleep(0.3)

if in_progress_found:
    print(f"\nListing API: found {in_progress_found} additional in-progress piece jobs.")
else:
    print(f"\nListing API done.")

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
