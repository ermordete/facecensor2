#!/usr/bin/python3
"""
effects.py - Alle Zensur-Effekte für FaceCensor Pro
=====================================================
Verfügbare Effekte:
- blur_light       : Leichter Gaussian Blur
- blur_strong      : Starker Gaussian Blur
- blur_gaussian    : Standard Gaussian Blur (mittel)
- blur_box         : Box Blur (gleichmäßige Unschärfe)
- pixelate_light   : Leichte Pixelierung
- pixelate_strong  : Starke Pixelierung
- black_bar        : Schwarzer Balken (Censor Bar)
- oval_blur        : Weicher Oval-Maskenbereich
- emoji            : Emoji/Sticker Overlay

Jeder Effekt ist zustandslos (pure function) und nimmt:
    frame, x, y, w, h, strength → modifiziertes frame
"""

import cv2
import numpy as np
from typing import Optional, Dict, Any
import os
import logging

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Hilfsfunktionen
# ─────────────────────────────────────────────────────────────────────────────

def _clip_region(frame: np.ndarray, x: int, y: int, w: int, h: int):
    """Gibt geclippte Region-Koordinaten zurück."""
    fh, fw = frame.shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(fw, x + w)
    y2 = min(fh, y + h)
    return x1, y1, x2, y2


def _odd(n: int) -> int:
    """Stellt sicher, dass n ungerade und >= 1 ist (für Gaussian Blur)."""
    n = max(1, n)
    return n if n % 2 == 1 else n + 1


# ─────────────────────────────────────────────────────────────────────────────
# Blur-Effekte
# ─────────────────────────────────────────────────────────────────────────────

def effect_blur_light(frame: np.ndarray, x: int, y: int, w: int, h: int,
                      strength: float = 1.0) -> np.ndarray:
    """Leichter Gaussian Blur."""
    x1, y1, x2, y2 = _clip_region(frame, x, y, w, h)
    region = frame[y1:y2, x1:x2]
    if region.size == 0:
        return frame
    k = _odd(int(15 * strength))
    frame[y1:y2, x1:x2] = cv2.GaussianBlur(region, (k, k), 0)
    return frame


def effect_blur_strong(frame: np.ndarray, x: int, y: int, w: int, h: int,
                       strength: float = 1.0) -> np.ndarray:
    """Starker Gaussian Blur."""
    x1, y1, x2, y2 = _clip_region(frame, x, y, w, h)
    region = frame[y1:y2, x1:x2]
    if region.size == 0:
        return frame
    k = _odd(int(51 * strength))
    frame[y1:y2, x1:x2] = cv2.GaussianBlur(region, (k, k), 30)
    return frame


def effect_blur_gaussian(frame: np.ndarray, x: int, y: int, w: int, h: int,
                         strength: float = 1.0) -> np.ndarray:
    """Standard Gaussian Blur."""
    x1, y1, x2, y2 = _clip_region(frame, x, y, w, h)
    region = frame[y1:y2, x1:x2]
    if region.size == 0:
        return frame
    k = _odd(int(31 * strength))
    frame[y1:y2, x1:x2] = cv2.GaussianBlur(region, (k, k), 15)
    return frame


def effect_blur_box(frame: np.ndarray, x: int, y: int, w: int, h: int,
                    strength: float = 1.0) -> np.ndarray:
    """Box Blur (gleichmäßige Unschärfe)."""
    x1, y1, x2, y2 = _clip_region(frame, x, y, w, h)
    region = frame[y1:y2, x1:x2]
    if region.size == 0:
        return frame
    k = _odd(int(25 * strength))
    frame[y1:y2, x1:x2] = cv2.blur(region, (k, k))
    return frame


# ─────────────────────────────────────────────────────────────────────────────
# Pixelation-Effekte
# ─────────────────────────────────────────────────────────────────────────────

def effect_pixelate_light(frame: np.ndarray, x: int, y: int, w: int, h: int,
                          strength: float = 1.0) -> np.ndarray:
    """Leichte Pixelierung (ca. 20x20 Pixel-Blöcke)."""
    x1, y1, x2, y2 = _clip_region(frame, x, y, w, h)
    region = frame[y1:y2, x1:x2]
    if region.size == 0:
        return frame
    rh, rw = region.shape[:2]
    # Divisor bestimmt Blockgröße: höher = gröbere Pixel
    divisor = max(2, int(20 * strength))
    sh = max(1, rh // divisor)
    sw = max(1, rw // divisor)
    small = cv2.resize(region, (sw, sh), interpolation=cv2.INTER_LINEAR)
    frame[y1:y2, x1:x2] = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)
    return frame


def effect_pixelate_strong(frame: np.ndarray, x: int, y: int, w: int, h: int,
                            strength: float = 1.0) -> np.ndarray:
    """Starke Pixelierung (ca. 8x8 Pixel-Blöcke)."""
    x1, y1, x2, y2 = _clip_region(frame, x, y, w, h)
    region = frame[y1:y2, x1:x2]
    if region.size == 0:
        return frame
    rh, rw = region.shape[:2]
    divisor = max(2, int(8 * strength))
    sh = max(1, rh // divisor)
    sw = max(1, rw // divisor)
    small = cv2.resize(region, (sw, sh), interpolation=cv2.INTER_LINEAR)
    frame[y1:y2, x1:x2] = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)
    return frame


# ─────────────────────────────────────────────────────────────────────────────
# Spezial-Effekte
# ─────────────────────────────────────────────────────────────────────────────

def effect_black_bar(frame: np.ndarray, x: int, y: int, w: int, h: int,
                     strength: float = 1.0) -> np.ndarray:
    """
    Schwarzer Balken (klassischer Censor Bar).
    Höhe des Balkens = 35% der Gesichtshöhe (Augenbereich).
    """
    x1, y1, x2, y2 = _clip_region(frame, x, y, w, h)
    if x2 <= x1 or y2 <= y1:
        return frame

    region_h = y2 - y1
    # Balken über die Augenhöhe legen (oberes Drittel)
    bar_height = max(10, int(region_h * 0.35))
    bar_y_start = y1 + int(region_h * 0.2)
    bar_y_end = min(y2, bar_y_start + bar_height)

    frame[bar_y_start:bar_y_end, x1:x2] = (0, 0, 0)
    return frame


def effect_oval_blur(frame: np.ndarray, x: int, y: int, w: int, h: int,
                     strength: float = 1.0) -> np.ndarray:
    """
    Weicher Oval/Ellipsen-Blur mit Soft Mask.
    Sieht professioneller aus als ein harter Rechteck-Blur.
    """
    x1, y1, x2, y2 = _clip_region(frame, x, y, w, h)
    region = frame[y1:y2, x1:x2].copy()
    if region.size == 0:
        return frame

    rh, rw = region.shape[:2]

    # Stark geblurrte Version
    k = _odd(int(51 * strength))
    blurred = cv2.GaussianBlur(region, (k, k), 30)

    # Ovale Maske erstellen
    mask = np.zeros((rh, rw), dtype=np.uint8)
    cv2.ellipse(
        mask,
        center=(rw // 2, rh // 2),
        axes=(rw // 2, rh // 2),
        angle=0,
        startAngle=0,
        endAngle=360,
        color=255,
        thickness=-1,  # Gefüllt
    )

    # Maske weichzeichnen für sanften Übergang
    mask = cv2.GaussianBlur(mask, (_odd(rw // 4), _odd(rh // 4)), 0)
    mask_3ch = cv2.merge([mask, mask, mask]).astype(float) / 255.0

    # Original und Blur mischen
    result = (blurred * mask_3ch + region * (1.0 - mask_3ch)).astype(np.uint8)
    frame[y1:y2, x1:x2] = result
    return frame


# ─────────────────────────────────────────────────────────────────────────────
# Emoji-Overlay
# ─────────────────────────────────────────────────────────────────────────────

class EmojiOverlay:
    """
    Verwaltet Emoji-Overlays für das Gesichts-Zensur.
    
    Emojis werden als PNG mit Alpha-Kanal geladen.
    Fallback: Farbige Rechtecke mit Text (falls keine PNG vorhanden).
    
    Erweiterbar: Einfach neue PNG-Dateien in emojis/ ablegen.
    """

    EMOJI_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "emojis")

    # Verfügbare Emojis: Name → Dateiname oder Unicode-Fallback
    AVAILABLE_EMOJIS = {
        "😎 Cool":    ("cool.png",    (70, 130, 180),  "COOL"),
        "😂 Lol":     ("lol.png",     (255, 200, 0),   "LOL"),
        "👻 Ghost":   ("ghost.png",   (200, 200, 200), "BOO"),
        "🤖 Robot":   ("robot.png",   (100, 200, 100), "BOT"),
        "🐱 Cat":     ("cat.png",     (255, 165, 0),   "CAT"),
        "🔥 Fire":    ("fire.png",    (255, 80, 0),    "FIRE"),
        "⭐ Star":    ("star.png",    (255, 215, 0),   "STAR"),
        "❓ Unknown": ("unknown.png", (150, 100, 200), "???"),
    }

    def __init__(self):
        self._cache: Dict[str, Optional[np.ndarray]] = {}
        self._load_emojis()

    def _load_emojis(self):
        """Lädt alle verfügbaren Emoji-PNGs."""
        os.makedirs(self.EMOJI_DIR, exist_ok=True)
        for name, (filename, color, text) in self.AVAILABLE_EMOJIS.items():
            path = os.path.join(self.EMOJI_DIR, filename)
            if os.path.exists(path):
                img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                if img is not None and img.shape[2] == 4:
                    self._cache[name] = img
                    logger.debug(f"Emoji geladen: {name}")
                    continue
            # Fallback: Synthetisches Emoji generieren
            self._cache[name] = self._make_fallback_emoji(color, text)

    def _make_fallback_emoji(
        self, color: tuple, text: str, size: int = 128
    ) -> np.ndarray:
        """
        Erstellt ein einfaches synthetisches Emoji-Bild als Fallback.
        Farbiger Kreis mit Text und Alpha-Kanal.
        """
        img = np.zeros((size, size, 4), dtype=np.uint8)

        # Gefüllter Kreis
        center = (size // 2, size // 2)
        radius = size // 2 - 4
        cv2.circle(img, center, radius, (*color, 230), -1)
        # Umrandung
        cv2.circle(img, center, radius, (255, 255, 255, 200), 2)

        # Text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.5, size / 128 * 0.8)
        thickness = max(1, size // 64)
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        tx = (size - text_size[0]) // 2
        ty = (size + text_size[1]) // 2
        cv2.putText(img, text, (tx, ty), font, font_scale, (255, 255, 255, 255), thickness)

        return img

    def get_names(self):
        return list(self.AVAILABLE_EMOJIS.keys())

    def apply(
        self,
        frame: np.ndarray,
        x: int, y: int, w: int, h: int,
        emoji_name: str,
        strength: float = 1.0,
    ) -> np.ndarray:
        """Legt Emoji über die Gesichtsregion."""
        emoji_img = self._cache.get(emoji_name)
        if emoji_img is None:
            return frame

        x1, y1, x2, y2 = _clip_region(frame, x, y, w, h)
        rw, rh = x2 - x1, y2 - y1
        if rw <= 0 or rh <= 0:
            return frame

        # Emoji auf Zielgröße skalieren
        resized = cv2.resize(emoji_img, (rw, rh), interpolation=cv2.INTER_AREA)

        if resized.shape[2] == 4:
            # Alpha-Compositing
            alpha = resized[:, :, 3:4].astype(float) / 255.0 * strength
            rgb = resized[:, :, :3].astype(float)
            bg = frame[y1:y2, x1:x2].astype(float)
            composite = rgb * alpha + bg * (1.0 - alpha)
            frame[y1:y2, x1:x2] = composite.astype(np.uint8)
        else:
            frame[y1:y2, x1:x2] = resized[:, :, :3]

        return frame


# ─────────────────────────────────────────────────────────────────────────────
# Effect Registry
# ─────────────────────────────────────────────────────────────────────────────

# Singleton für Emoji-Overlay
_emoji_overlay = EmojiOverlay()


def get_emoji_overlay() -> EmojiOverlay:
    return _emoji_overlay


# Alle nicht-Emoji-Effekte als Dictionary
EFFECTS = {
    "Blur Light":      effect_blur_light,
    "Blur Strong":     effect_blur_strong,
    "Blur Gaussian":   effect_blur_gaussian,
    "Blur Box":        effect_blur_box,
    "Pixel Light":     effect_pixelate_light,
    "Pixel Strong":    effect_pixelate_strong,
    "Black Bar":       effect_black_bar,
    "Oval Blur":       effect_oval_blur,
}

# Preset-Definitionen für schnelle Auswahl
PRESETS = {
    "Interview":   {"effect": "Blur Gaussian", "strength": 0.8},
    "Street":      {"effect": "Pixel Strong",  "strength": 1.0},
    "Streaming":   {"effect": "Blur Strong",   "strength": 0.9},
    "Funny":       {"effect": "emoji",         "strength": 1.0, "emoji": "😂 Lol"},
    "Safe Mode":   {"effect": "Black Bar",     "strength": 1.0},
    "Creator":     {"effect": "Oval Blur",     "strength": 0.85},
}


def apply_effect(
    frame: np.ndarray,
    x: int, y: int, w: int, h: int,
    effect_name: str,
    strength: float = 1.0,
    emoji_name: Optional[str] = None,
) -> np.ndarray:
    """
    Wendet den gewählten Effekt auf eine Gesichtsregion an.
    Zentraler Einstiegspunkt für alle Effekte.
    """
    if effect_name == "emoji":
        if emoji_name:
            return _emoji_overlay.apply(frame, x, y, w, h, emoji_name, strength)
        # Fallback auf Blur wenn kein Emoji ausgewählt
        return effect_blur_gaussian(frame, x, y, w, h, strength)

    effect_fn = EFFECTS.get(effect_name)
    if effect_fn:
        return effect_fn(frame, x, y, w, h, strength)

    # Unbekannter Effekt → Gaussian Blur als sicherer Fallback
    return effect_blur_gaussian(frame, x, y, w, h, strength)
