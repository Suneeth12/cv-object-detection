# 🖼️ Computer Vision — Object Detection & Segmentation

Real CV *evaluation* beyond a toy classifier: locate objects with bounding boxes
(scored by **mAP@0.5**) and label every pixel (scored by **mean IoU**). The
detector and segmenter are deliberately classical (sliding window + a small
classifier) so the project runs with no GPU — because the hard, transferable
part is the **detection/segmentation metrics**, which are identical for a CNN.

## What this project demonstrates
- **The detection pipeline** — window scoring → confidence threshold →
  **non-maximum suppression** → IoU matching against ground truth.
- **mAP from scratch** — precision-recall over ranked detections with IoU≥0.5
  matching and 11-point interpolated Average Precision.
- **Semantic segmentation + IoU** — per-pixel features → classification →
  Jaccard score against the true mask.
- **Honest difficulty** — objects sit in real noise, so segmentation IoU is
  0.97 (not a suspicious 1.0) and detection mAP reflects genuine misses.

## Demo

```text
$ python3 src/detection.py
mAP@0.5 = 0.775  over 120 images

$ python3 src/segmentation.py
mean IoU = 0.973  over 120 images
```

`reports/detections.png` overlays predicted boxes (orange) on ground truth
(green); `reports/segmentation.png` shows image / true mask / predicted mask.

## Components

| Piece | Where | Idea |
|---|---|---|
| Data | `src/generate_data.py` | noisy images + boxes + masks |
| Detection | `src/detection.py` | sliding window, NMS, IoU, mAP |
| Segmentation | `src/segmentation.py` | per-pixel features, mean IoU |

## Project structure
```
cv-object-detection/
├── data/images.npz        # images, boxes, masks (generated)
├── src/
│   ├── generate_data.py
│   ├── detection.py
│   └── segmentation.py
├── reports/               # detections.png, segmentation.png, *_metrics.json
├── requirements.txt
├── torun.txt
└── license.md
```

## Run it
```bash
./run.sh        # or see torun.txt
```

**Production swap**: replace the window scorer with a YOLO / Faster R-CNN head
and the pixel classifier with a U-Net / Mask R-CNN in PyTorch — the NMS, IoU
matching, mAP, and Jaccard evaluation code carries over unchanged.
