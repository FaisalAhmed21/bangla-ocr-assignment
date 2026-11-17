"""Shared preprocessing used by BOTH train.py and app.py.

Keeping this in one place guarantees training and inference see identical
inputs (a preprocessing mismatch is the #1 cause of bad predictions).

Convention: "ink" images are uint8, ink = bright (white), background = 0.
"""
import cv2
import numpy as np


def to_ink(gray: np.ndarray) -> np.ndarray:
    """Dataset images: make ink bright on a dark background.

    Handwriting datasets are normally dark ink on a light page, so if the
    image is mostly bright we invert it.
    """
    gray = gray.astype(np.uint8)
    if gray.mean() > 127:
        gray = 255 - gray
    return gray


def crop_square(ink: np.ndarray, size: int, margin: float = 0.12) -> np.ndarray:
    """Tight-crop the ink, centre it on a square canvas, resize to size x size."""
    _, mask = cv2.threshold(ink, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return np.zeros((size, size), np.uint8)
    crop = ink[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]
    h, w = crop.shape
    side = int(max(h, w) * (1 + 2 * margin)) + 2
    canvas = np.zeros((side, side), np.uint8)
    y0, x0 = (side - h) // 2, (side - w) // 2
    canvas[y0: y0 + h, x0: x0 + w] = crop
    return cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)
