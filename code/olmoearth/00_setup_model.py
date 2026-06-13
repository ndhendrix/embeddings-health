import json
import os
from pathlib import Path

import requests
from dotenv import load_dotenv

HERE = Path(__file__).parent
load_dotenv(HERE / ".env")

CONFIG = HERE / "config.json"

if CONFIG.exists():
    print("config.json already exists — delete it to re-run setup.")
    raise SystemExit(0)

BASE_URL = "https://olmoearth.allenai.org"
HEADERS = {
    "Authorization": f"Bearer {os.environ['OLMOEARTH_API_KEY']}",
    "Content-Type": "application/json",
}
PROJECT_ID = os.environ["OLMOEARTH_PROJECT_ID"]

resp = requests.post(
    f"{BASE_URL}/api/v1/models/from_config",
    json={
        "model_type": "embeddings",
        "wizard_answers": {
            "wizard_id": "embeddings_v1",
            "project_id": PROJECT_ID,
            "model_name": "ri-tract-embeddings-2022",
            "encoder_variant": "tiny",
            "resolution": "forty_meter",
            "num_periods": 12,
            "imagery_sources": ["sentinel2_l2a"],
        },
    },
    headers=HEADERS,
)
resp.raise_for_status()
model_id = resp.json()["records"][0]["id"]

CONFIG.write_text(json.dumps({"project_id": PROJECT_ID, "model_id": model_id}, indent=2))
print(f"Model created: {model_id}")
print(f"Written to {CONFIG}")
