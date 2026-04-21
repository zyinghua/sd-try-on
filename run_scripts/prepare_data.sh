#!/bin/bash
# Download + extract VITON-HD from Kaggle, then resize to 512x384 in place.
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python dataset/download.py
python dataset/resize.py
