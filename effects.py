#!/usr/bin/python3
"""
effects.py – Zensur-Effekte (vereinfacht)
==========================================
Einziger Effekt: Gaussian Blur für Gesichter / Hände.
Alle anderen Effekte (Pixelate, Black Bar, Oval Blur, Emoji) wurden entfernt.
"""

import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Hilfsfunktionen
# ─────────────────────────────────────────────────────────────────────────────

def _clip_region(frame: np.ndarray, x: int, y: int, w: int, h: int):
    """Gibt geclippte Region-Koordinaten zurück (verhindert Out-of-Bounds)."""
    fh, fw = frame.shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(fw, x + w)
    y2 = min(fh, y + h)
    return x1, y1, x2, y2


def _odd(n: int) -> int:
    """Stellt sicher, dass n ungerade und >= 1 ist (Pflicht für GaussianBlur)."""
    n = max(1, n)
    return n if n % 2 == 1 else n + 1


# ─────────────────────────────────────────────────────────────────────────────
# Einziger Effekt: Gaussian Blur
# ─────────────────────────────────────────────────────────────────────────────

def apply_blur(
    frame: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    strength: float = 1.0,
) -> np.ndarray:
    """
    Wendet Gaussian Blur auf eine rechteckige Region an.

    strength: 0.0 bis 1.0
      0.1 = sehr leicht  (Kernel ~9)
      0.5 = mittel       (Kernel ~27)
      1.0 = sehr stark   (Kernel ~51)

    PERFORMANCE: In-place-Operation auf dem Frame-Array,
    keine extra Kopie des gesamten Frames nötig.
    """
    x1, y1, x2, y2 = _clip_region(frame, x, y, w, h)
    if x2 <= x1 or y2 <= y1:
        return frame

    region = frame[y1:y2, x1:x2]
    if region.size == 0:
        return frame

    # Kernel-Größe skaliert linear mit strength: 5 (sehr schwach) bis 51 (sehr stark)
    k = _odd(int(5 + 46 * strength))
    frame[y1:y2, x1:x2] = cv2.GaussianBlur(region, (k, k), 0)
    return frame
