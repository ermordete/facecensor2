#!/usr/bin/python3
"""
detector.py - Gesichtserkennung mit temporalem Smoothing
=========================================================
VERBESSERUNGEN gegenüber Original:
- OpenCV DNN Face Detector statt Haar Cascade (deutlich besser + schneller)
- Haar Cascade als Fallback (falls kein Modell vorhanden)
- Temporales Smoothing der Bounding Boxes → kein Flackern
- Confidence-Schwellenwert einstellbar
- Face-Padding für saubere Abdeckung
- Multi-Face sauber implementiert

WICHTIG für Jetson Nano:
- DNN-Detector läuft auf CPU (kein CUDA nötig)
- SSD MobileNet ist sehr leichtgewichtig
- Haar Cascade als zuverlässiger Fallback
"""

import cv2
import numpy as np
import os
import logging
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

# Typ-Alias für Bounding Box
BBox = Tuple[int, int, int, int]  # (x, y, w, h)


# ─────────────────────────────────────────────────────────────────────────────
# Smoothing-Klasse für einzelnes Gesicht
# ─────────────────────────────────────────────────────────────────────────────

class TrackedFace:
    """
    Verfolgt ein einzelnes Gesicht über mehrere Frames.
    Exponentielles Moving Average (EMA) verhindert Flackern.
    """

    def __init__(self, bbox: BBox, alpha: float = 0.4):
        """
        alpha: Glättungsfaktor (0=sehr glatt/träge, 1=kein Smoothing)
        0.4 ist ein guter Kompromiss aus Reaktionsschnelligkeit und Stabilität
        """
        self.bbox = np.array(bbox, dtype=float)
        self.alpha = alpha
        self.missed_frames = 0  # Wie viele Frames wurde kein Match gefunden?
        self.max_missed = 8     # Nach N Frames ohne Match → Gesicht entfernen

    def update(self, bbox: BBox):
        """EMA-Update der Position."""
        new = np.array(bbox, dtype=float)
        # Exponentielles Moving Average
        self.bbox = self.alpha * new + (1 - self.alpha) * self.bbox
        self.missed_frames = 0

    def mark_missed(self):
        """Frame ohne Detection → zählen."""
        self.missed_frames += 1

    def is_valid(self) -> bool:
        return self.missed_frames < self.max_missed

    def get_bbox(self) -> BBox:
        """Gibt geglättete Bounding Box als Integer zurück."""
        return tuple(int(v) for v in self.bbox)


# ─────────────────────────────────────────────────────────────────────────────
# Basis-Detector-Klasse
# ─────────────────────────────────────────────────────────────────────────────

class FaceDetectorBase:
    """Basisklasse für alle Face Detectors."""

    def detect(self, frame: np.ndarray) -> List[BBox]:
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
# DNN-basierter Detektor (empfohlen)
# ─────────────────────────────────────────────────────────────────────────────

class DNNFaceDetector(FaceDetectorBase):
    """
    OpenCV DNN Face Detector (SSD + ResNet-10).
    
    VORTEILE gegenüber Haar Cascade:
    - Deutlich weniger False Positives
    - Erkennt Gesichter unter verschiedenen Winkeln
    - Confidence-Score verfügbar
    - Ähnliche Performance wie Haar Cascade auf CPU
    
    MODELL: deploy.prototxt + res10_300x300_ssd_iter_140000.caffemodel
    Download-Befehle im README.
    """

    MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    PROTOTXT = "deploy.prototxt"
    CAFFEMODEL = "res10_300x300_ssd_iter_140000.caffemodel"

    def __init__(self, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
        self.net = None
        self._load_model()

    def _load_model(self):
        prototxt_path = os.path.join(self.MODEL_DIR, self.PROTOTXT)
        model_path = os.path.join(self.MODEL_DIR, self.CAFFEMODEL)

        if not os.path.exists(prototxt_path) or not os.path.exists(model_path):
            logger.warning(
                f"DNN-Modell nicht gefunden unter {self.MODEL_DIR}. "
                "Fallback auf Haar Cascade. Siehe README für Download-Anleitung."
            )
            return

        try:
            self.net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
            # Nutze OpenCL wenn verfügbar (leichte GPU-Beschleunigung)
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            logger.info("✅ DNN Face Detector geladen")
        except Exception as e:
            logger.error(f"DNN-Modell konnte nicht geladen werden: {e}")
            self.net = None

    def is_available(self) -> bool:
        return self.net is not None

    def detect(self, frame: np.ndarray) -> List[BBox]:
        if self.net is None:
            return []

        h, w = frame.shape[:2]

        # PERFORMANCE: Resize auf 300x300 für DNN (Standard für SSD)
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            scalefactor=1.0,
            size=(300, 300),
            mean=(104.0, 177.0, 123.0),  # Mean Subtraction für dieses Modell
        )
        self.net.setInput(blob)
        detections = self.net.forward()

        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence < self.confidence_threshold:
                continue

            # Koordinaten von relativen zu absoluten umrechnen
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)

            # Clipping auf Frame-Grenzen
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            bw, bh = x2 - x1, y2 - y1
            if bw > 10 and bh > 10:  # Mindesgröße
                faces.append((x1, y1, bw, bh))

        return faces


# ─────────────────────────────────────────────────────────────────────────────
# Haar Cascade Detektor (Fallback)
# ─────────────────────────────────────────────────────────────────────────────

class HaarFaceDetector(FaceDetectorBase):
    """
    Haar Cascade Detektor als Fallback.
    Optimierte Parameter für weniger False Positives.
    """

    CASCADE_PATHS = [
        '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
        '/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml',
        '/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
        # OpenCV Python Package Pfad
        os.path.join(
            os.path.dirname(cv2.__file__),
            'data', 'haarcascade_frontalface_default.xml'
        ),
    ]

    def __init__(self):
        self.cascade = None
        self._load_cascade()

    def _load_cascade(self):
        for path in self.CASCADE_PATHS:
            if os.path.exists(path):
                cascade = cv2.CascadeClassifier(path)
                if not cascade.empty():
                    self.cascade = cascade
                    logger.info(f"✅ Haar Cascade geladen: {path}")
                    return
        logger.error("❌ Haar Cascade nicht gefunden!")

    def is_available(self) -> bool:
        return self.cascade is not None

    def detect(self, frame: np.ndarray) -> List[BBox]:
        if self.cascade is None:
            return []

        # PERFORMANCE: Auf Graustufen reduzieren (schneller)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Optional: Histogramm-Equalisierung für bessere Erkennung bei schlechtem Licht
        gray = cv2.equalizeHist(gray)

        faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor=1.15,   # Kleiner = mehr Schritte = besser aber langsamer
            minNeighbors=6,     # Höher = weniger False Positives
            minSize=(40, 40),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        if len(faces) == 0:
            return []
        return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]


# ─────────────────────────────────────────────────────────────────────────────
# Haupt-FaceDetector mit Smoothing und Tracking
# ─────────────────────────────────────────────────────────────────────────────

class FaceDetector:
    """
    Haupt-Gesichtsdetektor mit:
    - Automatischer Auswahl (DNN > Haar Cascade)
    - Temporalem Smoothing (EMA)
    - IoU-basiertem Tracking über Frames
    - Face Padding
    - Einstellbarer Confidence
    """

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        padding: float = 0.15,  # 15% Padding um Bounding Box
        smooth_alpha: float = 0.4,
        # PERFORMANCE: Detection nicht jeden Frame (jedes 2. Frame)
        detect_every_n_frames: int = 2,
    ):
        self.padding = padding
        self.detect_every_n_frames = detect_every_n_frames
        self._frame_counter = 0
        self._tracked_faces: List[TrackedFace] = []

        # Detector-Auswahl: DNN bevorzugt, Haar als Fallback
        dnn = DNNFaceDetector(confidence_threshold=confidence_threshold)
        if dnn.is_available():
            self._detector = dnn
            self.detector_name = "DNN (SSD)"
            logger.info("🎯 Nutze DNN Face Detector")
        else:
            haar = HaarFaceDetector()
            if haar.is_available():
                self._detector = haar
                self.detector_name = "Haar Cascade"
                logger.info("🎯 Nutze Haar Cascade (Fallback)")
            else:
                self._detector = None
                self.detector_name = "Nicht verfügbar"
                logger.error("❌ Kein Face Detector verfügbar!")

        self.smooth_alpha = smooth_alpha

    def _iou(self, a: BBox, b: BBox) -> float:
        """Intersection-over-Union für zwei Bounding Boxes."""
        ax, ay, aw, ah = a
        bx, by, bw, bh = b

        ix = max(ax, bx)
        iy = max(ay, by)
        iw = min(ax + aw, bx + bw) - ix
        ih = min(ay + ah, by + bh) - iy

        if iw <= 0 or ih <= 0:
            return 0.0

        intersection = iw * ih
        union = aw * ah + bw * bh - intersection
        return intersection / union if union > 0 else 0.0

    def _apply_padding(self, bbox: BBox, frame_w: int, frame_h: int) -> BBox:
        """Erweitert Bounding Box um Padding-Faktor."""
        x, y, w, h = bbox
        pad_x = int(w * self.padding)
        pad_y = int(h * self.padding)

        x_new = max(0, x - pad_x)
        y_new = max(0, y - pad_y)
        w_new = min(frame_w - x_new, w + 2 * pad_x)
        h_new = min(frame_h - y_new, h + 2 * pad_y)

        return (x_new, y_new, w_new, h_new)

    def detect(self, frame: np.ndarray) -> List[BBox]:
        """
        Hauptmethode: Erkennt Gesichter mit Smoothing.
        
        PERFORMANCE: Detection läuft nur jeden N-ten Frame.
        Zwischen den Frames werden die letzten bekannten Positionen verwendet.
        """
        if self._detector is None:
            return []

        h, w = frame.shape[:2]
        self._frame_counter += 1

        # Detection nur jeden N-ten Frame für Performance
        if self._frame_counter % self.detect_every_n_frames == 0:
            # PERFORMANCE: Auf kleinere Auflösung skalieren für Detection
            scale = 1.0
            detect_frame = frame
            if w > 640:
                scale = 640.0 / w
                detect_frame = cv2.resize(frame, (640, int(h * scale)))

            raw_detections = self._detector.detect(detect_frame)

            # Zurückskalieren falls nötig
            if scale != 1.0:
                raw_detections = [
                    (int(x / scale), int(y / scale), int(bw / scale), int(bh / scale))
                    for (x, y, bw, bh) in raw_detections
                ]

            # Tracking: Neue Detections mit bestehenden Tracks matchen
            matched = set()
            new_tracked = []

            for tracked in self._tracked_faces:
                best_iou = 0.3  # Minimaler IoU für Match
                best_det = None

                for i, det in enumerate(raw_detections):
                    if i in matched:
                        continue
                    iou = self._iou(tracked.get_bbox(), det)
                    if iou > best_iou:
                        best_iou = iou
                        best_det = i

                if best_det is not None:
                    tracked.update(raw_detections[best_det])
                    matched.add(best_det)
                    new_tracked.append(tracked)
                else:
                    tracked.mark_missed()
                    if tracked.is_valid():
                        new_tracked.append(tracked)

            # Neue Gesichter hinzufügen
            for i, det in enumerate(raw_detections):
                if i not in matched:
                    new_tracked.append(TrackedFace(det, alpha=self.smooth_alpha))

            self._tracked_faces = new_tracked

        # Padding anwenden und Bounding Boxes zurückgeben
        result = []
        for tracked in self._tracked_faces:
            bbox = tracked.get_bbox()
            padded = self._apply_padding(bbox, w, h)
            result.append(padded)

        return result

    def reset(self):
        """Tracking zurücksetzen."""
        self._tracked_faces = []
        self._frame_counter = 0

    @property
    def face_count(self) -> int:
        return len(self._tracked_faces)
