#!/usr/bin/env bash
# Generate images, run the sliding-window detector and the segmenter.
set -e
cd "$(dirname "$0")"

pip install -r requirements.txt
python3 src/generate_data.py
python3 src/detection.py
python3 src/segmentation.py
echo ""
echo "See reports/detections.png, reports/segmentation.png and the *_metrics.json files"
