"""Download the High-Resolution VITON Zalando dataset from Kaggle.

Requires Kaggle API credentials. Either:
  1. Fill in kaggle.json.template, rename it to kaggle.json, and place it
     in ~/.kaggle/ (or keep it next to this script — it will be loaded).
  2. Export KAGGLE_USERNAME and KAGGLE_KEY env vars.
"""

import importlib.util
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

DATASET = "marquis03/high-resolution-viton-zalando-dataset"
SCRIPT_DIR = Path(__file__).resolve().parent
DEST = SCRIPT_DIR / "data"


def ensure_kaggle():
    if importlib.util.find_spec("kaggle") is None:
        print("Installing kaggle...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle"])


def load_local_credentials():
    local = SCRIPT_DIR / "kaggle.json"
    if local.exists() and not (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")):
        with open(local) as f:
            creds = json.load(f)
        os.environ["KAGGLE_USERNAME"] = creds["username"]
        os.environ["KAGGLE_KEY"] = creds["key"]


def download():
    ensure_kaggle()
    load_local_credentials()
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    DEST.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {DATASET} to {DEST}...")
    api.dataset_download_files(DATASET, path=str(DEST), unzip=False, quiet=False)

    zip_name = DATASET.split("/")[-1] + ".zip"
    zip_path = DEST / zip_name
    if not zip_path.exists():
        candidates = list(DEST.glob("*.zip"))
        if not candidates:
            raise FileNotFoundError("No zip file found after download.")
        zip_path = candidates[0]

    print(f"Extracting {zip_path.name}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(DEST)

    zip_path.unlink()
    print(f"Done. Dataset available at {DEST}")


if __name__ == "__main__":
    download()
