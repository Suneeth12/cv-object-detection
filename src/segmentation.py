"""Semantic segmentation by per-pixel classification, scored by mean IoU.

For every pixel we build a tiny local feature vector (its intensity plus the
mean/std/max of its 3×3 neighborhood) and train a classifier to label it
foreground vs background. Predicted masks are scored against ground truth with
the intersection-over-union (Jaccard) metric — the standard segmentation score.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def features(img):
    """Per-pixel features via shifted views (intensity + local stats)."""
    H, W = img.shape
    pad = np.pad(img, 1, mode="edge")
    nbrs = np.stack([pad[i:i + H, j:j + W]
                     for i in range(3) for j in range(3)], axis=-1)  # (H,W,9)
    feats = np.concatenate([
        img[..., None],
        nbrs.mean(-1, keepdims=True),
        nbrs.std(-1, keepdims=True),
        nbrs.max(-1, keepdims=True),
    ], axis=-1)
    return feats.reshape(-1, feats.shape[-1])


def iou_mask(pred, gt):
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return 1.0 if union == 0 else inter / union


def main():
    d = np.load(ROOT / "data" / "images.npz", allow_pickle=True)
    Xtr, Mtr, Xte, Mte = d["Xtr"], d["Mtr"], d["Xte"], d["Mte"]

    # train on a subset of pixels (sample to keep it fast & balanced-ish)
    rng = np.random.default_rng(3)
    FX, FY = [], []
    for img, mask in zip(Xtr[:120], Mtr[:120]):
        f = features(img); y = mask.ravel()
        idx = rng.choice(len(y), 400, replace=False)
        FX.append(f[idx]); FY.append(y[idx])
    clf = RandomForestClassifier(n_estimators=120, max_depth=8, random_state=0)
    clf.fit(np.vstack(FX), np.concatenate(FY))

    ious = []
    preds = []
    for img, mask in zip(Xte, Mte):
        p = clf.predict(features(img)).reshape(img.shape).astype(bool)
        preds.append(p); ious.append(iou_mask(p, mask.astype(bool)))
    miou = float(np.mean(ious))

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "segmentation_metrics.json").write_text(json.dumps(
        {"mean_IoU": round(miou, 3), "test_images": len(Xte)}, indent=2))

    fig, axes = plt.subplots(3, 4, figsize=(11, 8))
    for col in range(4):
        axes[0, col].imshow(Xte[col], cmap="gray"); axes[0, col].set_title("image")
        axes[1, col].imshow(Mte[col], cmap="gray"); axes[1, col].set_title("ground truth")
        axes[2, col].imshow(preds[col], cmap="gray")
        axes[2, col].set_title(f"pred IoU={ious[col]:.2f}")
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"Semantic segmentation — mean IoU = {miou:.3f}")
    fig.tight_layout(); fig.savefig(REPORTS / "segmentation.png", dpi=110)

    print(f"mean IoU = {miou:.3f}  over {len(Xte)} images")
    print("See reports/segmentation.png and reports/segmentation_metrics.json")


if __name__ == "__main__":
    main()
