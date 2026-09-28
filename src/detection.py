"""Sliding-window object detector with NMS, scored by mAP@0.5.

Pipeline (classical, from scratch):
  1. train a patch classifier (object-centered vs background) on raw patch pixels
  2. at test, slide a window over the image and score every location
  3. threshold + non-maximum suppression (NMS) to get final boxes
  4. match to ground truth by IoU≥0.5 and compute Average Precision (mAP)

The emphasis is the *detection evaluation* (IoU matching, precision-recall, AP)
— the part that's identical whether the scorer is logistic regression or a CNN.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
PS = 12        # patch size
STRIDE = 2
SCORE_THRESH = 0.5
NMS_IOU = 0.2


def iou(a, b):
    r0 = max(a[0], b[0]); c0 = max(a[1], b[1])
    r1 = min(a[2], b[2]); c1 = min(a[3], b[3])
    inter = max(0, r1 - r0) * max(0, c1 - c0)
    if inter == 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua


def patches(img, boxes, rng, n_pos_jit=2):
    """Build labelled training patches: positives centered on objects, plus negatives."""
    H, W = img.shape
    X, y = [], []
    occupied = []
    for (r0, c0, r1, c1) in boxes:
        cr, cc = (r0 + r1) // 2, (c0 + c1) // 2
        for _ in range(n_pos_jit):
            jr, jc = rng.integers(-1, 2), rng.integers(-1, 2)
            tr, tc = cr - PS // 2 + jr, cc - PS // 2 + jc
            if 0 <= tr <= H - PS and 0 <= tc <= W - PS:
                X.append(img[tr:tr + PS, tc:tc + PS].ravel()); y.append(1)
        occupied.append((cr, cc))
    for _ in range(len(boxes) * 4 + 3):
        tr, tc = rng.integers(0, H - PS), rng.integers(0, W - PS)
        cr, cc = tr + PS // 2, tc + PS // 2
        if all(abs(cr - orr) + abs(cc - occ) > PS for orr, occ in occupied):
            X.append(img[tr:tr + PS, tc:tc + PS].ravel()); y.append(0)
    return X, y


def detect(img, clf):
    H, W = img.shape
    dets = []
    for r in range(0, H - PS + 1, STRIDE):
        for c in range(0, W - PS + 1, STRIDE):
            p = clf.predict_proba(img[r:r + PS, c:c + PS].ravel()[None, :])[0, 1]
            if p >= SCORE_THRESH:
                dets.append((p, [r, c, r + PS, c + PS]))
    dets.sort(reverse=True, key=lambda d: d[0])
    keep = []
    for score, box in dets:
        if all(iou(box, kb) < NMS_IOU for _, kb in keep):
            keep.append((score, box))
    return keep


def average_precision(all_dets, all_gts):
    """AP@0.5 over the test set via the precision-recall curve."""
    total_gt = sum(len(g) for g in all_gts)
    flat = []
    for i, dets in enumerate(all_dets):
        for score, box in dets:
            flat.append((score, i, box))
    flat.sort(reverse=True, key=lambda x: x[0])
    matched = {i: set() for i in range(len(all_gts))}
    tp, fp = [], []
    for score, i, box in flat:
        gts = all_gts[i]
        best, best_j = 0.0, -1
        for j, g in enumerate(gts):
            if j in matched[i]:
                continue
            v = iou(box, g)
            if v > best:
                best, best_j = v, j
        if best >= 0.5:
            matched[i].add(best_j); tp.append(1); fp.append(0)
        else:
            tp.append(0); fp.append(1)
    tp = np.cumsum(tp); fp = np.cumsum(fp)
    recall = tp / max(total_gt, 1)
    precision = tp / np.maximum(tp + fp, 1)
    # 11-point interpolated AP
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        p = precision[recall >= t].max() if np.any(recall >= t) else 0.0
        ap += p / 11
    return float(ap), recall, precision


def main():
    d = np.load(ROOT / "data" / "images.npz", allow_pickle=True)
    Xtr, Btr = d["Xtr"], d["Btr"]
    Xte, Bte = d["Xte"], d["Bte"]
    rng = np.random.default_rng(1)

    PX, PY = [], []
    for img, boxes in zip(Xtr, Btr):
        xs, ys = patches(img, boxes, rng)
        PX += xs; PY += ys
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(np.array(PX), np.array(PY))

    all_dets = [detect(img, clf) for img in Xte]
    all_gts = [list(map(list, b)) for b in Bte]
    ap, recall, precision = average_precision(all_dets, all_gts)

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "detection_metrics.json").write_text(json.dumps(
        {"mAP@0.5": round(ap, 3),
         "patch_classifier_train_acc": round(clf.score(np.array(PX), np.array(PY)), 3),
         "test_images": len(Xte),
         "total_test_objects": int(sum(len(g) for g in all_gts))}, indent=2))

    # visualize detections on a few images
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))
    for ax, img, dets, gts in zip(axes, Xte, all_dets, all_gts):
        ax.imshow(img, cmap="gray")
        for g in gts:
            ax.add_patch(plt.Rectangle((g[1], g[0]), g[3] - g[1], g[2] - g[0],
                         fill=False, edgecolor="#2a9d8f", lw=2))
        for s, b in dets:
            ax.add_patch(plt.Rectangle((b[1], b[0]), b[3] - b[1], b[2] - b[0],
                         fill=False, edgecolor="#e76f51", lw=1, ls="--"))
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Detections (orange dashed) vs ground truth (green) — mAP@0.5 = %.2f" % ap)
    fig.tight_layout(); fig.savefig(REPORTS / "detections.png", dpi=110)

    print(f"mAP@0.5 = {ap:.3f}  over {len(Xte)} images")
    print("See reports/detections.png and reports/detection_metrics.json")


if __name__ == "__main__":
    main()
