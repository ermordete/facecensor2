#!/usr/bin/python3
"""
ui.py – Hauptfenster FaceCensor Pro (vereinfacht)
==================================================
Kernfunktionen:
  1. Live-Kamerafeed mit Gesichts-Blur
  2. Aufnahme starten / stoppen
  3. Screenshot speichern
  4. Blur-Stärke Slider
  5. Handzensur Ein/Aus
  6. Gesichtsmodell wechseln (Haar / DNN)

Farbschema:
  Hintergrund : #1F2A36  (Navy Blue)
  Buttons     : #E1DACA  (Chalk Beige)
  Text        : #CBCCBE  (Sage)
  Akzent      : #4A8FD4  (Blau – aktive Zustände)
  Aufnahme    : #C94F4F  (Rot – laufende Aufnahme)
"""

import cv2
import numpy as np
import logging
import time
from collections import deque
from typing import Optional

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QSlider,
    QGroupBox, QSizePolicy, QStatusBar,
    QComboBox, QFrame,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QFont

from camera import CameraCapture
from detector import Detector, FACE_MODELS, FACE_MODEL_A
from effects import apply_blur
from recorder import VideoRecorder, save_screenshot

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Zentrales Stylesheet
# ─────────────────────────────────────────────────────────────────────────────

QSS = """
/* ── Basis ── */
* {
    font-family: "Noto Sans", "DejaVu Sans", "Liberation Sans", "Ubuntu", sans-serif;
    font-size: 13px;
}

QMainWindow, QWidget {
    background-color: #1F2A36;
    color: #CBCCBE;
}

/* ── Buttons ── */
QPushButton {
    background-color: #E1DACA;
    color: #1F2A36;
    border: none;
    border-radius: 6px;
    padding: 9px 16px;
    font-weight: 600;
    font-size: 12px;
    min-height: 38px;
}
QPushButton:hover {
    background-color: #EDE9E0;
}
QPushButton:pressed {
    background-color: #CCC9BC;
}
QPushButton:disabled {
    background-color: #2E3F52;
    color: #5A6A7A;
}
QPushButton[active="true"] {
    background-color: #4A8FD4;
    color: #FFFFFF;
}
QPushButton[active="true"]:hover {
    background-color: #5A9FE4;
}
QPushButton[recording="true"] {
    background-color: #C94F4F;
    color: #FFFFFF;
    font-weight: 700;
}
QPushButton[recording="true"]:hover {
    background-color: #D96060;
}

/* ── Gruppenboxen / Panels ── */
QGroupBox {
    background-color: #253241;
    border: 1px solid #2E3F52;
    border-radius: 8px;
    margin-top: 18px;
    padding-top: 10px;
    padding-left: 10px;
    padding-right: 10px;
    padding-bottom: 10px;
    color: #8A9AA8;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    top: 1px;
    padding: 0 4px;
    text-transform: uppercase;
}

/* ── Slider ── */
QSlider::groove:horizontal {
    height: 4px;
    background: #2E3F52;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #E1DACA;
    width: 16px;
    height: 16px;
    margin: -6px 0;
    border-radius: 8px;
    border: none;
}
QSlider::handle:horizontal:hover {
    background: #EDE9E0;
}
QSlider::sub-page:horizontal {
    background: #4A8FD4;
    border-radius: 2px;
}

/* ── ComboBox ── */
QComboBox {
    background-color: #E1DACA;
    color: #1F2A36;
    border: none;
    border-radius: 6px;
    padding: 7px 10px;
    min-height: 34px;
    font-weight: 600;
    font-size: 12px;
}
QComboBox:hover {
    background-color: #EDE9E0;
}
QComboBox::drop-down {
    border: none;
    width: 22px;
}
QComboBox::down-arrow {
    width: 10px;
    height: 10px;
}
QComboBox QAbstractItemView {
    background-color: #253241;
    color: #CBCCBE;
    border: 1px solid #2E3F52;
    selection-background-color: #4A8FD4;
    selection-color: #FFFFFF;
    padding: 4px;
}

/* ── Labels ── */
QLabel {
    color: #CBCCBE;
}
QLabel[role="dimmed"] {
    color: #6A7A8A;
    font-size: 11px;
}
QLabel[role="stat-value"] {
    color: #E1DACA;
    font-size: 22px;
    font-weight: 700;
}
QLabel[role="stat-label"] {
    color: #6A7A8A;
    font-size: 10px;
    letter-spacing: 1px;
}
QLabel[role="title"] {
    color: #CBCCBE;
    font-size: 18px;
    font-weight: 700;
    letter-spacing: 0.5px;
}
QLabel[role="subtitle"] {
    color: #6A7A8A;
    font-size: 10px;
    letter-spacing: 2px;
}

/* ── Statusleiste ── */
QStatusBar {
    background-color: #253241;
    color: #6A7A8A;
    border-top: 1px solid #2E3F52;
    font-size: 11px;
    padding: 3px 10px;
}

/* ── Trennlinie ── */
QFrame[frameShape="4"],
QFrame[frameShape="HLine"] {
    color: #2E3F52;
    max-height: 1px;
    background: #2E3F52;
}
"""


# ─────────────────────────────────────────────────────────────────────────────
# Processing-Thread
# ─────────────────────────────────────────────────────────────────────────────

class ProcessingThread(QThread):
    """
    Läuft getrennt vom UI-Thread.
    Liest Kamera-Frames, erkennt Gesichter/Hände, wendet Blur an,
    sendet das fertige Frame per Signal an die UI.

    FPS-Berechnung: Gleitender Durchschnitt über 1 Sekunde.
    """

    # Signal: (frame_bgr, gesichter_count, haende_count, fps)
    frame_ready = pyqtSignal(np.ndarray, int, int, float)

    def __init__(self, camera: CameraCapture, detector: Detector):
        super().__init__()
        self.camera   = camera
        self.detector = detector
        self._running = False

        # Einstellungen (aus UI-Thread gesetzt, gelesen im Processing-Thread)
        self.strength: float = 0.7

        # FPS-Messung: deque mit Zeitstempeln der letzten Frames
        self._ts_queue: deque = deque(maxlen=30)

    def run(self):
        self._running = True
        while self._running:
            ret, frame = self.camera.read()
            if not ret or frame is None:
                self.msleep(8)
                continue

            # Zeitstempel für FPS-Berechnung
            now = time.monotonic()
            self._ts_queue.append(now)

            # Gesichts- und Handerkennung
            face_boxes, hand_boxes = self.detector.detect(frame)

            # Blur auf alle Boxen anwenden
            s = self.strength
            for (x, y, w, h) in face_boxes:
                frame = apply_blur(frame, x, y, w, h, s)
            for (x, y, w, h) in hand_boxes:
                frame = apply_blur(frame, x, y, w, h, s)

            # FPS: Frames in letzter Sekunde zählen
            cutoff = now - 1.0
            recent = [t for t in self._ts_queue if t >= cutoff]
            fps = float(len(recent))

            self.frame_ready.emit(frame, len(face_boxes), len(hand_boxes), fps)

    def stop(self):
        self._running = False
        self.wait(2000)


# ─────────────────────────────────────────────────────────────────────────────
# Hilfsfunktionen
# ─────────────────────────────────────────────────────────────────────────────

def _separator() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.HLine)
    f.setProperty("frameShape", "4")
    return f


def _panel(title: str) -> QGroupBox:
    return QGroupBox(title)


def _set_prop(widget, key: str, value):
    """Setzt eine Qt-Property und aktualisiert den Style (für QSS-Selektoren)."""
    widget.setProperty(key, value)
    widget.style().unpolish(widget)
    widget.style().polish(widget)


# ─────────────────────────────────────────────────────────────────────────────
# Hauptfenster
# ─────────────────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("FaceCensor Pro")
        self.setMinimumSize(960, 620)
        self.setStyleSheet(QSS)

        # Komponenten
        self.camera   = CameraCapture(
            sensor_id=0,
            capture_width=1280, capture_height=720,
            display_width=640,  display_height=480,
            framerate=30,
        )
        self.detector = Detector(
            face_model=FACE_MODEL_A,
            detect_faces_every=2,   # [P1] Gesicht jede 2. Frame
            detect_hands_every=4,   # [P1] Hand jede 4. Frame
            face_padding=0.15,
            hand_padding=0.10,
            smooth_alpha=0.4,
            max_detect_width=480,   # [P2] Downscale auf 480px
        )
        self.recorder  = VideoRecorder(output_dir="aufnahmen")
        self.proc_thread: Optional[ProcessingThread] = None

        # Letzter Frame für Screenshot
        self._last_frame: Optional[np.ndarray] = None

        self._build_ui()
        self._setup_statusbar()

    # ── UI aufbauen ──────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        layout = QHBoxLayout(root)
        layout.setContentsMargins(16, 16, 16, 10)
        layout.setSpacing(14)

        left = self._build_left()
        left.setFixedWidth(200)
        layout.addWidget(left)

        center = self._build_center()
        layout.addWidget(center, stretch=1)

        right = self._build_right()
        right.setFixedWidth(200)
        layout.addWidget(right)

    # ── Linkes Panel ─────────────────────────────────────────────────────────

    def _build_left(self) -> QWidget:
        w = QWidget()
        vl = QVBoxLayout(w)
        vl.setContentsMargins(0, 0, 0, 0)
        vl.setSpacing(10)

        # App-Titel
        title = QLabel("FaceCensor")
        title.setProperty("role", "title")
        sub   = QLabel("PRO  ·  NANO EDITION")
        sub.setProperty("role", "subtitle")
        vl.addWidget(title)
        vl.addWidget(sub)
        vl.addWidget(_separator())

        # ── Kamera ──
        cam_panel = _panel("Kamera")
        cam_vl = QVBoxLayout(cam_panel)
        cam_vl.setSpacing(6)

        self.btn_start = QPushButton("Starten")
        self.btn_start.clicked.connect(self._on_start)
        cam_vl.addWidget(self.btn_start)

        self.btn_stop = QPushButton("Stoppen")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._on_stop)
        cam_vl.addWidget(self.btn_stop)

        vl.addWidget(cam_panel)

        # ── Modellauswahl ──
        model_panel = _panel("Erkennungsmodell")
        model_vl = QVBoxLayout(model_panel)
        model_vl.setSpacing(6)

        self.combo_model = QComboBox()
        self.combo_model.addItems(FACE_MODELS)
        self.combo_model.setCurrentText(FACE_MODEL_A)
        self.combo_model.currentTextChanged.connect(self._on_model_changed)
        model_vl.addWidget(self.combo_model)

        self.lbl_model_info = QLabel("Haar: schnell, kein Download\nDNN: genauer, Dateien nötig")
        self.lbl_model_info.setProperty("role", "dimmed")
        self.lbl_model_info.setWordWrap(True)
        model_vl.addWidget(self.lbl_model_info)

        vl.addWidget(model_panel)

        # ── Handzensur ──
        hand_panel = _panel("Handzensur")
        hand_vl = QVBoxLayout(hand_panel)
        hand_vl.setSpacing(6)

        self.btn_hands = QPushButton("Handzensur: Aus")
        self.btn_hands.setCheckable(True)
        self.btn_hands.clicked.connect(self._on_toggle_hands)
        hand_vl.addWidget(self.btn_hands)

        self.lbl_hand_info = QLabel("Benötigt:\nmodels/haarcascade_hand.xml")
        self.lbl_hand_info.setProperty("role", "dimmed")
        self.lbl_hand_info.setWordWrap(True)
        hand_vl.addWidget(self.lbl_hand_info)

        vl.addWidget(hand_panel)

        vl.addStretch()
        return w

    # ── Mittlerer Bereich (Video) ─────────────────────────────────────────────

    def _build_center(self) -> QWidget:
        w = QWidget()
        vl = QVBoxLayout(w)
        vl.setContentsMargins(0, 0, 0, 0)
        vl.setSpacing(8)

        # Video-Label
        self.video_lbl = QLabel()
        self.video_lbl.setAlignment(Qt.AlignCenter)
        self.video_lbl.setMinimumSize(480, 360)
        self.video_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_lbl.setStyleSheet(
            "background-color: #0D1620; "
            "border: 1px solid #2E3F52; "
            "border-radius: 8px; "
            "color: #3A5068; "
            "font-size: 14px;"
        )
        self.video_lbl.setText("Kamera nicht aktiv\n\nLinks auf Starten klicken")
        vl.addWidget(self.video_lbl, stretch=1)

        # Stats-Leiste
        stats_row = self._build_stats_row()
        vl.addWidget(stats_row)

        return w

    def _build_stats_row(self) -> QWidget:
        w = QWidget()
        w.setStyleSheet("background-color: #253241; border-radius: 6px;")
        hl = QHBoxLayout(w)
        hl.setContentsMargins(16, 6, 16, 6)
        hl.setSpacing(32)

        def stat_widget(label_text: str):
            container = QWidget()
            vl = QVBoxLayout(container)
            vl.setContentsMargins(0, 0, 0, 0)
            vl.setSpacing(1)
            val = QLabel("—")
            val.setProperty("role", "stat-value")
            val.setAlignment(Qt.AlignCenter)
            lbl = QLabel(label_text.upper())
            lbl.setProperty("role", "stat-label")
            lbl.setAlignment(Qt.AlignCenter)
            vl.addWidget(val)
            vl.addWidget(lbl)
            return container, val

        fps_w,   self.lbl_fps     = stat_widget("Bildrate")
        face_w,  self.lbl_faces   = stat_widget("Gesichter")
        hand_w,  self.lbl_hands   = stat_widget("Hände")
        status_w, self.lbl_status_val = stat_widget("Status")

        hl.addWidget(fps_w)
        hl.addWidget(face_w)
        hl.addWidget(hand_w)
        hl.addWidget(status_w)

        self.lbl_status_val.setText("Bereit")
        return w

    # ── Rechtes Panel ────────────────────────────────────────────────────────

    def _build_right(self) -> QWidget:
        w = QWidget()
        vl = QVBoxLayout(w)
        vl.setContentsMargins(0, 0, 0, 0)
        vl.setSpacing(10)

        # ── Blur-Stärke ──
        blur_panel = _panel("Blur-Stärke")
        blur_vl = QVBoxLayout(blur_panel)
        blur_vl.setSpacing(8)

        self.lbl_strength = QLabel("Stärke:  70 %")
        blur_vl.addWidget(self.lbl_strength)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(10, 100)
        self.slider.setValue(70)
        self.slider.valueChanged.connect(self._on_strength_changed)
        blur_vl.addWidget(self.slider)

        vl.addWidget(blur_panel)

        # ── Aufnahme ──
        rec_panel = _panel("Aufnahme")
        rec_vl = QVBoxLayout(rec_panel)
        rec_vl.setSpacing(6)

        self.btn_record = QPushButton("Aufnahme starten")
        self.btn_record.setEnabled(False)
        self.btn_record.clicked.connect(self._on_toggle_record)
        rec_vl.addWidget(self.btn_record)

        self.lbl_rec_file = QLabel("")
        self.lbl_rec_file.setProperty("role", "dimmed")
        self.lbl_rec_file.setWordWrap(True)
        rec_vl.addWidget(self.lbl_rec_file)

        vl.addWidget(rec_panel)

        # ── Screenshot ──
        shot_panel = _panel("Screenshot")
        shot_vl = QVBoxLayout(shot_panel)

        self.btn_shot = QPushButton("Screenshot speichern")
        self.btn_shot.setEnabled(False)
        self.btn_shot.clicked.connect(self._on_screenshot)
        shot_vl.addWidget(self.btn_shot)

        self.lbl_shot_file = QLabel("")
        self.lbl_shot_file.setProperty("role", "dimmed")
        self.lbl_shot_file.setWordWrap(True)
        shot_vl.addWidget(self.lbl_shot_file)

        vl.addWidget(shot_panel)

        vl.addStretch()
        return w

    # ── Statusleiste ─────────────────────────────────────────────────────────

    def _setup_statusbar(self):
        sb = QStatusBar()
        self.setStatusBar(sb)
        self._sb_label = QLabel("FaceCensor Pro  ·  Bereit")
        sb.addWidget(self._sb_label)

    def _status(self, msg: str):
        self._sb_label.setText(msg)

    # ── Kamera starten / stoppen ──────────────────────────────────────────────

    def _on_start(self):
        self._status("Kamera wird gestartet …")
        if not self.camera.start():
            self._status("Kamera nicht gefunden.")
            self.lbl_status_val.setText("Fehler")
            return

        self.proc_thread = ProcessingThread(self.camera, self.detector)
        self.proc_thread.strength = self.slider.value() / 100.0
        self.proc_thread.frame_ready.connect(self._on_frame)
        self.proc_thread.start()

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_record.setEnabled(True)
        self.btn_shot.setEnabled(True)
        self.lbl_status_val.setText("Live")
        self._status(f"Kamera aktiv  ·  Modell: {self.detector.face_model_name}")

    def _on_stop(self):
        if self.proc_thread:
            self.proc_thread.stop()
            self.proc_thread = None
        if self.recorder.is_recording:
            self.recorder.stop()
        self.camera.stop()

        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_record.setEnabled(False)
        self.btn_shot.setEnabled(False)
        self.video_lbl.setText("Kamera nicht aktiv\n\nLinks auf Starten klicken")
        self.lbl_fps.setText("—")
        self.lbl_faces.setText("—")
        self.lbl_hands.setText("—")
        self.lbl_status_val.setText("Bereit")
        self._status("Kamera gestoppt")
        _set_prop(self.btn_record, "recording", False)
        self.btn_record.setText("Aufnahme starten")

    # ── Frame empfangen und anzeigen ─────────────────────────────────────────

    def _on_frame(self, frame: np.ndarray, faces: int, hands: int, fps: float):
        """
        Slot: Empfängt fertigen Frame vom Processing-Thread.
        Konvertiert BGR → RGB → QPixmap und zeigt ihn an.
        PERFORMANCE: Nur eine Konvertierung, keine extra Kopien.
        """
        self._last_frame = frame

        if self.recorder.is_recording:
            self.recorder.write_frame(frame)

        # Stats aktualisieren
        self.lbl_fps.setText(f"{fps:.1f}")
        self.lbl_faces.setText(str(faces))
        self.lbl_hands.setText(str(hands) if self.detector.hands_enabled else "—")

        # BGR → RGB (OpenCV nutzt BGR, Qt erwartet RGB)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qt_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img).scaled(
            self.video_lbl.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.video_lbl.setPixmap(pixmap)

    # ── Slider ───────────────────────────────────────────────────────────────

    def _on_strength_changed(self, val: int):
        self.lbl_strength.setText(f"Stärke:  {val} %")
        if self.proc_thread:
            self.proc_thread.strength = val / 100.0

    # ── Modellauswahl ─────────────────────────────────────────────────────────

    def _on_model_changed(self, model_name: str):
        self.detector.set_face_model(model_name)
        self._status(f"Modell gewechselt: {model_name}")

    # ── Handzensur ───────────────────────────────────────────────────────────

    def _on_toggle_hands(self, checked: bool):
        if checked and not self.detector.hand_detector_available:
            self.btn_hands.setChecked(False)
            self._status(
                "Hand-Cascade nicht gefunden – "
                "bitte models/haarcascade_hand.xml ablegen."
            )
            return
        self.detector.hands_enabled = checked
        label = "Handzensur: Ein" if checked else "Handzensur: Aus"
        self.btn_hands.setText(label)
        _set_prop(self.btn_hands, "active", checked)
        self._status(label)

    # ── Aufnahme ──────────────────────────────────────────────────────────────

    def _on_toggle_record(self):
        if self.recorder.is_recording:
            fname = self.recorder.stop()
            self.btn_record.setText("Aufnahme starten")
            _set_prop(self.btn_record, "recording", False)
            self.lbl_rec_file.setText(fname or "")
            self.lbl_status_val.setText("Live")
            self._status(f"Aufnahme gespeichert: {fname}")
        else:
            fname = self.recorder.start(width=640, height=480, fps=25.0)
            if fname:
                self.btn_record.setText("Aufnahme stoppen")
                _set_prop(self.btn_record, "recording", True)
                self.lbl_rec_file.setText("")
                self.lbl_status_val.setText("REC")
                self._status(f"Aufnahme läuft: {fname}")
            else:
                self._status("Aufnahme konnte nicht gestartet werden.")

    # ── Screenshot ───────────────────────────────────────────────────────────

    def _on_screenshot(self):
        if self._last_frame is not None:
            fname = save_screenshot(self._last_frame, output_dir="screenshots")
            self.lbl_shot_file.setText(fname)
            self._status(f"Screenshot gespeichert: {fname}")
        else:
            self._status("Kein Frame verfügbar.")

    # ── Tastatur-Shortcuts ───────────────────────────────────────────────────

    def keyPressEvent(self, event):
        k = event.key()
        if k == Qt.Key_Escape:
            self.close()
        elif k == Qt.Key_Space:
            if self.camera.is_running():
                self._on_stop()
            else:
                self._on_start()
        elif k == Qt.Key_S and self.btn_shot.isEnabled():
            self._on_screenshot()
        elif k == Qt.Key_R and self.btn_record.isEnabled():
            self._on_toggle_record()

    # ── Sauberes Beenden ──────────────────────────────────────────────────────

    def closeEvent(self, event):
        if self.recorder.is_recording:
            self.recorder.stop()
        if self.proc_thread:
            self.proc_thread.stop()
        self.camera.stop()
        event.accept()
