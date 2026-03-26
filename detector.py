#!/usr/bin/python3
"""
detector.py – Gesichts- und Handerkennung
==========================================
Modell A (Gesicht): Haar Cascade     – sehr schnell, leichtgewichtig
Modell B (Gesicht): OpenCV DNN SSD   – genauer, etwas schwerer

Handerkennung: Haar Cascade (fist + open palm, kombiniert)
Alle Detektoren arbeiten performance-optimiert:
  - Detection nur alle N Frames
  - Downscale vor Detection, Rückskalierung danach
  - Temporales EMA-Smoothing der Bounding Boxes
"""

import cv2
import numpy as np
import os
import logging
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

BBox = Tuple[int, int, int, int]  # (x, y, w, h)


# ─────────────────────────────────────────────────────────────────────────────
# EMA-Tracking eines einzelnen Objekts
# ─────────────────────────────────────────────────────────────────────────────

class TrackedBox:
    """
    Verfolgt eine Bounding Box über Frames mit exponentiellem Moving Average.
    Verhindert Flackern / Springen der Zensur-Maske.
    """

    def __init__(self, bbox: BBox, alpha: float = 0.4):
        # alpha: 0 = sehr träge/glatt, 1 = kein Smoothing
        self.bbox = np.array(bbox, dtype=float)
        self.alpha = alpha
        self.missed = 0
        self.max_missed = 8  # Nach N Frames ohne Match entfernen

    def update(self, bbox: BBox):
        new = np.array(bbox, dtype=float)
        self.bbox = self.alpha * new + (1.0 - self.alpha) * self.bbox
        self.missed = 0

    def mark_missed(self):
        self.missed += 1

    def is_valid(self) -> bool:
        return self.missed < self.max_missed

    def get(self) -> BBox:
        return tuple(int(v) for v in self.bbox)


# ─────────────────────────────────────────────────────────────────────────────
# IoU-Hilfsfunktion
# ─────────────────────────────────────────────────────────────────────────────

def _iou(a: BBox, b: BBox) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(ax, bx)
    iy = max(ay, by)
    iw = min(ax + aw, bx + bw) - ix
    ih = min(ay + ah, by + bh) - iy
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Haar Cascade Gesichtsdetektor (Modell A – schnell)
# ─────────────────────────────────────────────────────────────────────────────

class HaarFaceDetector:
    """
    Modell A: OpenCV Haar Cascade.
    Sehr schnell, leichtgewichtig, ideal für Jetson Nano.
    Keine externen Modelldateien nötig (in OpenCV enthalten).
    """

    NAME = "Haar Cascade  (schnell)"

    CASCADE_PATHS = [
        '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
        '/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml',
        '/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
        os.path.join(
            os.path.dirname(cv2.__file__),
            'data', 'haarcascade_frontalface_default.xml'
        ),
    ]

    def __init__(self):
        self.cascade: Optional[cv2.CascadeClassifier] = None
        for path in self.CASCADE_PATHS:
            if os.path.exists(path):
                c = cv2.CascadeClassifier(path)
                if not c.empty():
                    self.cascade = c
                    logger.info(f"Haar Cascade geladen: {path}")
                    break
        if self.cascade is None:
            logger.error("Haar Cascade nicht gefunden!")

    def is_available(self) -> bool:
        return self.cascade is not None

    def detect(self, gray: np.ndarray) -> List[BBox]:
        """Erwartet Graustufen-Frame."""
        if self.cascade is None:
            return []
        faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor=1.15,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        if len(faces) == 0:
            return []
        return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]


# ─────────────────────────────────────────────────────────────────────────────
# DNN SSD Gesichtsdetektor (Modell B – genauer)
# ─────────────────────────────────────────────────────────────────────────────

class DNNFaceDetector:
    """
    Modell B: OpenCV DNN – SSD ResNet-10.
    Genauer als Haar Cascade, erkennt auch seitliche Gesichter.
    Benötigt Modelldateien im Ordner models/ (siehe README).

    Download:
      wget https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt
      wget https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel
    """

    NAME = "DNN SSD  (genauer)"

    MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    PROTOTXT   = "deploy.prototxt"
    CAFFEMODEL = "res10_300x300_ssd_iter_140000.caffemodel"

    def __init__(self, confidence: float = 0.5):
        self.confidence = confidence
        self.net = None

        proto = os.path.join(self.MODEL_DIR, self.PROTOTXT)
        model = os.path.join(self.MODEL_DIR, self.CAFFEMODEL)

        if os.path.exists(proto) and os.path.exists(model):
            try:
                self.net = cv2.dnn.readNetFromCaffe(proto, model)
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                logger.info("DNN Face Detector geladen")
            except Exception as e:
                logger.error(f"DNN laden fehlgeschlagen: {e}")
                self.net = None
        else:
            logger.warning(
                "DNN-Modell nicht gefunden. Bitte models/ befüllen (siehe README)."
            )

    def is_available(self) -> bool:
        return self.net is not None

    def detect_bgr(self, frame_bgr: np.ndarray) -> List[BBox]:
        """Erwartet BGR-Frame (Original)."""
        if self.net is None:
            return []
        h, w = frame_bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame_bgr, (300, 300)),
            1.0, (300, 300), (104.0, 177.0, 123.0),
        )
        self.net.setInput(blob)
        dets = self.net.forward()

        faces = []
        for i in range(dets.shape[2]):
            conf = dets[0, 0, i, 2]
            if conf < self.confidence:
                continue
            box = dets[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            bw, bh = x2 - x1, y2 - y1
            if bw > 10 and bh > 10:
                faces.append((x1, y1, bw, bh))
        return faces


# ─────────────────────────────────────────────────────────────────────────────
# Haar Cascade Hand-Detektor
# ─────────────────────────────────────────────────────────────────────────────

class HaarHandDetector:
    """
    Handerkennung via Haar Cascade.

    Verwendet fist- und palm-Cascades (falls vorhanden).
    Die Cascades müssen manuell heruntergeladen werden:

      models/haarcascade_hand.xml
      → https://github.com/Balaje/OpenCV/raw/master/haarcascades/haarcascade_hand.xml

    Wenn keine Hand-Cascade vorhanden ist, ist der Detektor deaktiviert
    (keine Exception, stille Deaktivierung).

    PERFORMANCE:
    - Wird seltener ausgeführt als Face Detection (alle 4 Frames)
    - Downscale auf 320px Breite vor Detection
    """

    NAME = "Haar Cascade (Hände)"

    # Mögliche Pfade für Hand-Cascades
    CASCADE_CANDIDATES = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "models", "haarcascade_hand.xml"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "models", "hand.xml"),
    ]

    def __init__(self):
        self.cascade: Optional[cv2.CascadeClassifier] = None
        for path in self.CASCADE_CANDIDATES:
            if os.path.exists(path):
                c = cv2.CascadeClassifier(path)
                if not c.empty():
                    self.cascade = c
                    logger.info(f"Hand-Cascade geladen: {path}")
                    break
        if self.cascade is None:
            logger.info(
                "Hand-Cascade nicht gefunden. "
                "Bitte models/haarcascade_hand.xml ablegen (siehe README)."
            )

    def is_available(self) -> bool:
        return self.cascade is not None

    def detect(self, gray: np.ndarray) -> List[BBox]:
        if self.cascade is None:
            return []
        hands = self.cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=4,
            minSize=(40, 40),
        )
        if len(hands) == 0:
            return []
        return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in hands]


# ─────────────────────────────────────────────────────────────────────────────
# Haupt-Detektor mit Tracking, Smoothing, Performance-Optimierung
# ─────────────────────────────────────────────────────────────────────────────

# Verfügbare Gesichtsmodelle für UI-Auswahl
FACE_MODEL_A = "Haar Cascade  (schnell)"
FACE_MODEL_B = "DNN SSD  (genauer)"
FACE_MODELS  = [FACE_MODEL_A, FACE_MODEL_B]


class Detector:
    """
    Zentraler Detektor für Gesichter und optional Hände.

    PERFORMANCE-MASSNAHMEN (kommentiert):
    - [P1] Detection nur alle N Frames → spart CPU
    - [P2] Downscale auf max. 480px Breite vor Detection
    - [P3] Bounding Boxes per EMA geglättet → kein Flackern
    - [P4] IoU-Tracking: neue Detektion mit letzter Box abgeglichen
    - [P5] Graustufen-Konvertierung einmalig pro Frame
    """

    def __init__(
        self,
        face_model: str = FACE_MODEL_A,
        detect_faces_every: int = 2,   # [P1] Gesicht: jede 2. Frame
        detect_hands_every: int = 4,   # [P1] Hand: jede 4. Frame
        face_padding: float = 0.15,    # 15% Padding um Face-Box
        hand_padding: float = 0.10,
        smooth_alpha: float = 0.4,     # [P3] EMA-Faktor
        max_detect_width: int = 480,   # [P2] Downscale-Zielbreite
        dnn_confidence: float = 0.5,
    ):
        self.face_padding = face_padding
        self.hand_padding = hand_padding
        self.detect_faces_every = detect_faces_every
        self.detect_hands_every = detect_hands_every
        self.max_detect_width = max_detect_width
        self.smooth_alpha = smooth_alpha

        # Gesichtsdetektoren (beide vorladen)
        self._haar = HaarFaceDetector()
        self._dnn  = DNNFaceDetector(confidence=dnn_confidence)

        # Handdetektor
        self._hand_det = HaarHandDetector()
        self.hands_enabled = False

        # Aktives Face-Modell
        self._face_model_name = face_model
        self._face_detector = self._resolve_face_detector(face_model)

        # Tracking-State
        self._face_tracks: List[TrackedBox] = []
        self._hand_tracks: List[TrackedBox] = []
        self._face_frame_counter = 0
        self._hand_frame_counter = 0

    def _resolve_face_detector(self, model_name: str):
        """Gibt den passenden Detektor-Callable zurück."""
        if model_name == FACE_MODEL_B and self._dnn.is_available():
            return self._dnn
        return self._haar  # Fallback auf Haar

    def set_face_model(self, model_name: str):
        """Gesichtsmodell zur Laufzeit wechseln."""
        self._face_model_name = model_name
        self._face_detector = self._resolve_face_detector(model_name)
        self._face_tracks = []  # Tracking nach Modellwechsel zurücksetzen
        logger.info(f"Face-Modell gewechselt: {model_name}")

    @property
    def face_model_name(self) -> str:
        return self._face_model_name

    @property
    def hand_detector_available(self) -> bool:
        return self._hand_det.is_available()

    def _scale_down(self, frame: np.ndarray):
        """
        [P2] Skaliert Frame für Detection herunter.
        Gibt (scale_factor, kleines_frame) zurück.
        """
        h, w = frame.shape[:2]
        if w <= self.max_detect_width:
            return 1.0, frame
        scale = self.max_detect_width / w
        small = cv2.resize(frame, (self.max_detect_width, int(h * scale)))
        return scale, small

    def _scale_boxes(self, boxes: List[BBox], scale: float) -> List[BBox]:
        """[P2] Skaliert Bounding Boxes zurück auf Original-Größe."""
        if scale == 1.0:
            return boxes
        return [
            (int(x / scale), int(y / scale), int(w / scale), int(h / scale))
            for (x, y, w, h) in boxes
        ]

    def _pad_box(self, bbox: BBox, fw: int, fh: int, pad: float) -> BBox:
        """Erweitert Box um Padding-Faktor, geclipt auf Frame-Grenzen."""
        x, y, w, h = bbox
        px, py = int(w * pad), int(h * pad)
        x0 = max(0, x - px)
        y0 = max(0, y - py)
        x1 = min(fw, x + w + px)
        y1 = min(fh, y + h + py)
        return (x0, y0, x1 - x0, y1 - y0)

    def _update_tracks(
        self,
        tracks: List[TrackedBox],
        detections: List[BBox],
    ) -> List[TrackedBox]:
        """
        [P3/P4] IoU-Tracking + EMA-Smoothing.
        Matcht neue Detektionen mit bestehenden Tracks.
        """
        matched = set()
        new_tracks = []

        for track in tracks:
            best_iou = 0.25
            best_idx = None
            for i, det in enumerate(detections):
                if i in matched:
                    continue
                score = _iou(track.get(), det)
                if score > best_iou:
                    best_iou = score
                    best_idx = i

            if best_idx is not None:
                track.update(detections[best_idx])
                matched.add(best_idx)
                new_tracks.append(track)
            else:
                track.mark_missed()
                if track.is_valid():
                    new_tracks.append(track)

        # Neue Detektionen ohne Match → neue Tracks anlegen
        for i, det in enumerate(detections):
            if i not in matched:
                new_tracks.append(TrackedBox(det, alpha=self.smooth_alpha))

        return new_tracks

    def detect(self, frame: np.ndarray):
        """
        Haupt-Methode: Gibt (face_boxes, hand_boxes) zurück.
        Beide Listen enthalten gepaddte, geglättete BBoxen.

        [P1] Detection läuft nur alle N Frames, dazwischen werden
             die letzten bekannten Tracks zurückgegeben.
        """
        fh, fw = frame.shape[:2]
        self._face_frame_counter += 1
        self._hand_frame_counter += 1

        # ── Gesichts-Detection ──────────────────────────────────────────────
        if self._face_frame_counter % self.detect_faces_every == 0:
            scale, small = self._scale_down(frame)  # [P2]

            if isinstance(self._face_detector, HaarFaceDetector):
                # [P5] Haar braucht Graustufen
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                raw = self._face_detector.detect(gray)
            else:
                # DNN arbeitet auf BGR
                raw = self._face_detector.detect_bgr(small)

            raw = self._scale_boxes(raw, scale)  # [P2]
            self._face_tracks = self._update_tracks(self._face_tracks, raw)  # [P3/P4]

        # ── Hand-Detection (optional) ────────────────────────────────────────
        if self.hands_enabled and self._hand_det.is_available():
            if self._hand_frame_counter % self.detect_hands_every == 0:
                scale, small = self._scale_down(frame)
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                raw_h = self._hand_det.detect(gray)
                raw_h = self._scale_boxes(raw_h, scale)
                self._hand_tracks = self._update_tracks(self._hand_tracks, raw_h)
        else:
            self._hand_tracks = []

        # ── Padding anwenden und zurückgeben ────────────────────────────────
        face_boxes = [
            self._pad_box(t.get(), fw, fh, self.face_padding)
            for t in self._face_tracks
        ]
        hand_boxes = [
            self._pad_box(t.get(), fw, fh, self.hand_padding)
            for t in self._hand_tracks
        ]
        return face_boxes, hand_boxes

    def reset(self):
        self._face_tracks = []
        self._hand_tracks = []
        self._face_frame_counter = 0
        self._hand_frame_counter = 0

    @property
    def face_count(self) -> int:
        return len(self._face_tracks)

    @property
    def hand_count(self) -> int:
        return len(self._hand_tracks)
