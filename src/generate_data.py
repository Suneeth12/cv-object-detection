"""Synthetic images with bright square objects on a noisy background.

Each image holds 1-3 axis-aligned square objects at random positions. We save,
per image: the pixels, the ground-truth bounding boxes, and a foreground mask.
This gives us exact labels to score detection (mAP@0.5) and segmentation (IoU)
without downloading a dataset.
"""
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
H = W = 48
N_TRAIN, N_TEST = 250, 120
OBJ_MIN, OBJ_MAX = 8, 13


def make_image(rng):
    img = rng.normal(0.25, 0.13, (H, W))
    mask = np.zeros((H, W), dtype=np.uint8)
    boxes = []
    k = rng.integers(1, 4)
    placed = 0
    for _ in range(k * 4):
        if placed >= k:
            break
        s = rng.integers(OBJ_MIN, OBJ_MAX)
        r0 = rng.integers(0, H - s); c0 = rng.integers(0, W - s)
        if mask[max(0, r0 - 3):r0 + s + 3, max(0, c0 - 3):c0 + s + 3].any():
            continue                      # keep objects from touching
        img[r0:r0 + s, c0:c0 + s] += rng.uniform(0.35, 0.55)
        mask[r0:r0 + s, c0:c0 + s] = 1
        boxes.append([r0, c0, r0 + s, c0 + s])
        placed += 1
    return np.clip(img, 0, 1), np.array(boxes), mask


def build(n, rng):
    imgs, boxes, masks = [], [], []
    for _ in range(n):
        im, bx, mk = make_image(rng)
        imgs.append(im); boxes.append(bx); masks.append(mk)
    return np.array(imgs), np.array(boxes, dtype=object), np.array(masks)


def main():
    DATA.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    Xtr, Btr, Mtr = build(N_TRAIN, rng)
    Xte, Bte, Mte = build(N_TEST, rng)
    np.savez(DATA / "images.npz", Xtr=Xtr, Btr=Btr, Mtr=Mtr,
             Xte=Xte, Bte=Bte, Mte=Mte, allow_pickle=True)
    tot = sum(len(b) for b in Bte)
    print(f"Saved {N_TRAIN} train / {N_TEST} test images ({tot} test objects) -> data/images.npz")


if __name__ == "__main__":
    main()
