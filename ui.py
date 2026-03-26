#!/usr/bin/python3
"""
ui.py - Hauptfenster / GUI für FaceCensor Pro
================================================
Technologie: PyQt5
Layout: Kamerafeed (Mitte) | Steuerung (Links + Rechts)
Farbschema:
  - Background: #1F2A36 (Navy Blue)
  - Buttons:    #E1DACA (Chalk Beige)
  - Text:       #CBCCBE (Sage)
  - Akzent:     #4A90D9 (Hellblau für aktive Elemente)
  - Recording:  #E05050 (Rot für aktive Aufnahme)
"""

import cv2
import numpy as np
import logging
import os
from typing import Optional

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QSlider, QComboBox,
    QGroupBox, QSizePolicy, QSpacerItem, QStatusBar,
    QFrame, QScrollArea,
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor, QPalette, QIcon

from camera import CameraCapture
from detector import FaceDetector
from effects import EFFECTS, PRESETS, apply_effect, get_emoji_overlay
from recorder import VideoRecorder, save_screenshot

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Farb-Konstanten
# ─────────────────────────────────────────────────────────────────────────────

C_BG          = "#1F2A36"   # Navy Blue – Haupthintergrund
C_PANEL       = "#253241"   # Etwas heller für Panels
C_BTN         = "#E1DACA"   # Chalk Beige – Buttons
C_BTN_HOVER   = "#F0EDE4"   # Heller Hover-State
C_BTN_PRESS   = "#C8C4B5"   # Dunkler Press-State
C_BTN_ACTIVE  = "#4A90D9"   # Blau – aktiver Button
C_BTN_ACTIVE_HOVER = "#5EA3EF"
C_TEXT        = "#CBCCBE"   # Sage – Text
C_TEXT_DIM    = "#8A9AA8"   # Gedimmter Text
C_ACCENT      = "#4A90D9"   # Blau – Akzent
C_RECORD      = "#E05050"   # Rot – Aufnahme
C_RECORD_HOVER = "#EF6060"
C_BORDER      = "#2E3F52"   # Dezente Umrandung


# ─────────────────────────────────────────────────────────────────────────────
# Globales Stylesheet
# ─────────────────────────────────────────────────────────────────────────────

GLOBAL_STYLE = f"""
/* ── Allgemeines ── */
QMainWindow, QWidget {{
    background-color: {C_BG};
    color: {C_TEXT};
    font-family: "Segoe UI", "Ubuntu", "Helvetica Neue", sans-serif;
    font-size: 13px;
}}

/* ── Standard-Button (Chalk Beige) ── */
QPushButton {{
    background-color: {C_BTN};
    color: {C_BG};
    border: none;
    border-radius: 8px;
    padding: 8px 14px;
    font-weight: 600;
    font-size: 12px;
    min-height: 36px;
}}
QPushButton:hover {{
    background-color: {C_BTN_HOVER};
    /* Sanfter Glow-Effekt */
    border: 1px solid rgba(255,255,255,0.3);
}}
QPushButton:pressed {{
    background-color: {C_BTN_PRESS};
    padding-top: 9px;
    padding-bottom: 7px;
}}
QPushButton:disabled {{
    background-color: #3A4A5A;
    color: {C_TEXT_DIM};
}}

/* ── Aktiver / ausgewählter Button (Blau) ── */
QPushButton#active {{
    background-color: {C_BTN_ACTIVE};
    color: white;
}}
QPushButton#active:hover {{
    background-color: {C_BTN_ACTIVE_HOVER};
}}

/* ── Recording-Button (Rot) ── */
QPushButton#record {{
    background-color: {C_RECORD};
    color: white;
    font-weight: 700;
}}
QPushButton#record:hover {{
    background-color: {C_RECORD_HOVER};
}}

/* ── GroupBox (Panel/Card) ── */
QGroupBox {{
    background-color: {C_PANEL};
    border: 1px solid {C_BORDER};
    border-radius: 10px;
    margin-top: 16px;
    padding: 8px;
    font-weight: 700;
    font-size: 11px;
    color: {C_TEXT_DIM};
    letter-spacing: 1px;
    text-transform: uppercase;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    top: 0px;
    padding: 0 4px;
}}

/* ── Slider ── */
QSlider::groove:horizontal {{
    height: 4px;
    background: {C_BORDER};
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {C_BTN};
    border: none;
    width: 16px;
    height: 16px;
    margin: -6px 0;
    border-radius: 8px;
}}
QSlider::handle:horizontal:hover {{
    background: {C_BTN_HOVER};
}}
QSlider::sub-page:horizontal {{
    background: {C_ACCENT};
    border-radius: 2px;
}}

/* ── ComboBox ── */
QComboBox {{
    background-color: {C_PANEL};
    color: {C_TEXT};
    border: 1px solid {C_BORDER};
    border-radius: 6px;
    padding: 6px 10px;
    min-height: 32px;
}}
QComboBox::drop-down {{
    border: none;
    width: 24px;
}}
QComboBox QAbstractItemView {{
    background-color: {C_PANEL};
    color: {C_TEXT};
    border: 1px solid {C_BORDER};
    selection-background-color: {C_ACCENT};
    selection-color: white;
    padding: 4px;
}}

/* ── Status Bar ── */
QStatusBar {{
    background-color: {C_PANEL};
    color: {C_TEXT_DIM};
    border-top: 1px solid {C_BORDER};
    font-size: 11px;
    padding: 4px 10px;
}}

/* ── Label ── */
QLabel#section_title {{
    color: {C_TEXT_DIM};
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    padding-top: 4px;
}}
QLabel#stat_value {{
    color: {C_TEXT};
    font-size: 18px;
    font-weight: 700;
}}
QLabel#stat_label {{
    color: {C_TEXT_DIM};
    font-size: 10px;
}}
"""


# ─────────────────────────────────────────────────────────────────────────────
# Hilfsfunktionen / Widgets
# ─────────────────────────────────────────────────────────────────────────────

def make_separator() -> QFrame:
    """Horizontale Trennlinie."""
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setStyleSheet(f"color: {C_BORDER}; background: {C_BORDER}; max-height: 1px;")
    return line


def make_section_label(text: str) -> QLabel:
    """Kleines Abschnitts-Label."""
    lbl = QLabel(text.upper())
    lbl.setObjectName("section_title")
    return lbl


def make_panel(title: str) -> QGroupBox:
    """Erstellt eine Panel/Card-Gruppe."""
    box = QGroupBox(title)
    return box


# ─────────────────────────────────────────────────────────────────────────────
# Processing-Thread (Frame-Verarbeitung außerhalb des UI-Threads)
# ─────────────────────────────────────────────────────────────────────────────

class ProcessingThread(QThread):
    """
    Verarbeitungs-Thread für Gesichtserkennung + Effekte.
    
    PERFORMANCE: Läuft komplett außerhalb des UI-Threads.
    Sendet fertig verarbeitete Frames via Signal an die UI.
    """

    frame_ready = pyqtSignal(np.ndarray, int, float)  # frame, face_count, fps

    def __init__(self, camera: CameraCapture, detector: FaceDetector):
        super().__init__()
        self.camera = camera
        self.detector = detector
        self._running = False

        # Aktuelle Einstellungen (thread-safe via einfache Zuweisung für primitives)
        self.effect_name = "Blur Gaussian"
        self.strength = 1.0
        self.emoji_name: Optional[str] = None
        self.show_overlay = True  # Debug-Overlay (FPS, Gesichter etc.)

    def run(self):
        self._running = True
        while self._running:
            ret, frame = self.camera.read()
            if not ret or frame is None:
                self.msleep(10)
                continue

            # Gesichter erkennen
            faces = self.detector.detect(frame)

            # Effekte anwenden
            for (x, y, w, h) in faces:
                frame = apply_effect(
                    frame, x, y, w, h,
                    self.effect_name,
                    self.strength,
                    self.emoji_name,
                )

            # Signal an UI senden
            self.frame_ready.emit(frame, len(faces), self.camera.fps)

    def stop(self):
        self._running = False
        self.wait(2000)


# ─────────────────────────────────────────────────────────────────────────────
# Haupt-Fenster
# ─────────────────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    """Hauptfenster der FaceCensor Pro Applikation."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("FaceCensor Pro  ·  Jetson Nano Edition")
        self.setMinimumSize(1100, 700)

        # Komponenten initialisieren
        self.camera = CameraCapture(
            sensor_id=0,
            capture_width=1280,
            capture_height=720,
            display_width=640,
            display_height=480,
            framerate=30,
        )
        self.detector = FaceDetector(
            confidence_threshold=0.5,
            padding=0.15,
            smooth_alpha=0.4,
            detect_every_n_frames=2,
        )
        self.recorder = VideoRecorder(output_dir="recordings")
        self.proc_thread: Optional[ProcessingThread] = None

        # UI aufbauen
        self.setStyleSheet(GLOBAL_STYLE)
        self._build_ui()
        self._setup_status_bar()

        # Statistiken
        self._last_face_count = 0
        self._last_fps = 0.0

    def _build_ui(self):
        """Baut das gesamte UI auf."""
        central = QWidget()
        self.setCentralWidget(central)

        # Haupt-Layout: [Links-Panel] [Video-Feed] [Rechts-Panel]
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(16, 16, 16, 8)
        root_layout.setSpacing(16)

        # ── Linkes Panel ──
        left_panel = self._build_left_panel()
        left_panel.setFixedWidth(220)
        root_layout.addWidget(left_panel)

        # ── Video-Feed (Mitte) ──
        video_container = self._build_video_area()
        root_layout.addWidget(video_container, stretch=1)

        # ── Rechtes Panel ──
        right_panel = self._build_right_panel()
        right_panel.setFixedWidth(220)
        root_layout.addWidget(right_panel)

    # ── Linkes Panel ────────────────────────────────────────────────────────

    def _build_left_panel(self) -> QWidget:
        """Links: Kamera-Steuerung + Effekt-Auswahl + Presets."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        # App-Titel
        title = QLabel("FaceCensor")
        title.setStyleSheet(
            f"color: {C_TEXT}; font-size: 20px; font-weight: 700; "
            f"font-style: italic; letter-spacing: 1px;"
        )
        subtitle = QLabel("Pro  ·  Nano Edition")
        subtitle.setStyleSheet(
            f"color: {C_TEXT_DIM}; font-size: 10px; letter-spacing: 2px; margin-bottom: 8px;"
        )
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addWidget(make_separator())

        # ── Kamera-Sektion ──
        cam_box = make_panel("Camera")
        cam_layout = QVBoxLayout(cam_box)
        cam_layout.setSpacing(8)

        self.btn_start = QPushButton("▶  Start Camera")
        self.btn_start.clicked.connect(self._on_start_camera)
        cam_layout.addWidget(self.btn_start)

        self.btn_stop = QPushButton("■  Stop Camera")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._on_stop_camera)
        cam_layout.addWidget(self.btn_stop)

        layout.addWidget(cam_box)

        # ── Effekt-Sektion ──
        fx_box = make_panel("Effect")
        fx_layout = QVBoxLayout(fx_box)
        fx_layout.setSpacing(6)

        # Effekt-Buttons
        self._effect_buttons = {}
        effects_order = [
            ("Blur Light",    "Blur  ·  Light"),
            ("Blur Strong",   "Blur  ·  Strong"),
            ("Blur Gaussian", "Blur  ·  Gaussian"),
            ("Blur Box",      "Blur  ·  Box"),
            ("Pixel Light",   "Pixel  ·  Light"),
            ("Pixel Strong",  "Pixel  ·  Strong"),
            ("Black Bar",     "Black Bar"),
            ("Oval Blur",     "Oval Blur"),
        ]
        for key, label in effects_order:
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.clicked.connect(lambda checked, k=key: self._on_select_effect(k))
            fx_layout.addWidget(btn)
            self._effect_buttons[key] = btn

        # Standardmäßig Gaussian aktiv
        self._active_effect = "Blur Gaussian"
        self._update_effect_buttons()

        layout.addWidget(fx_box)

        # ── Emoji-Sektion ──
        emoji_box = make_panel("Emoji")
        emoji_layout = QVBoxLayout(emoji_box)

        self.btn_emoji_toggle = QPushButton("😎  Emoji Mode")
        self.btn_emoji_toggle.setCheckable(True)
        self.btn_emoji_toggle.clicked.connect(self._on_toggle_emoji)
        emoji_layout.addWidget(self.btn_emoji_toggle)

        self.emoji_combo = QComboBox()
        emoji_overlay = get_emoji_overlay()
        self.emoji_combo.addItems(emoji_overlay.get_names())
        self.emoji_combo.currentTextChanged.connect(self._on_emoji_changed)
        emoji_layout.addWidget(self.emoji_combo)
        self.emoji_combo.setEnabled(False)

        layout.addWidget(emoji_box)

        layout.addStretch()
        return panel

    # ── Video-Bereich (Mitte) ────────────────────────────────────────────────

    def _build_video_area(self) -> QWidget:
        """Mitte: Video-Feed + Statusleiste darunter."""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Video-Label
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet(
            f"background-color: #0D1620; "
            f"border: 1px solid {C_BORDER}; "
            f"border-radius: 10px;"
        )
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding
        )
        layout.addWidget(self.video_label, stretch=1)

        # Platzhalter-Text wenn Kamera nicht aktiv
        self.video_label.setText(
            "Camera not started\n\nPress  ▶ Start Camera  to begin"
        )
        self.video_label.setStyleSheet(
            f"background-color: #0D1620; "
            f"border: 1px solid {C_BORDER}; "
            f"border-radius: 10px; "
            f"color: {C_TEXT_DIM}; "
            f"font-size: 16px;"
        )

        # Stats-Leiste unter dem Video
        stats_row = QWidget()
        stats_row.setStyleSheet(
            f"background-color: {C_PANEL}; border-radius: 8px; padding: 4px;"
        )
        stats_layout = QHBoxLayout(stats_row)
        stats_layout.setContentsMargins(12, 4, 12, 4)
        stats_layout.setSpacing(24)

        def make_stat(value_id: str, label_text: str):
            w = QWidget()
            vl = QVBoxLayout(w)
            vl.setContentsMargins(0, 0, 0, 0)
            vl.setSpacing(0)
            val = QLabel("–")
            val.setObjectName("stat_value")
            val.setAlignment(Qt.AlignCenter)
            lbl = QLabel(label_text)
            lbl.setObjectName("stat_label")
            lbl.setAlignment(Qt.AlignCenter)
            vl.addWidget(val)
            vl.addWidget(lbl)
            return w, val

        fps_w, self.lbl_fps    = make_stat("fps",    "FPS")
        face_w, self.lbl_faces = make_stat("faces",  "Faces")
        mode_w, self.lbl_mode  = make_stat("mode",   "Mode")
        rec_w, self.lbl_rec    = make_stat("rec",    "Status")

        stats_layout.addWidget(fps_w)
        stats_layout.addWidget(face_w)
        stats_layout.addWidget(mode_w)
        stats_layout.addWidget(rec_w)

        layout.addWidget(stats_row)
        return container

    # ── Rechtes Panel ────────────────────────────────────────────────────────

    def _build_right_panel(self) -> QWidget:
        """Rechts: Stärke-Slider + Presets + Recording + Screenshot."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        # ── Stärke-Slider ──
        strength_box = make_panel("Strength")
        strength_layout = QVBoxLayout(strength_box)

        self.lbl_strength = QLabel("Intensity:  100%")
        self.lbl_strength.setStyleSheet(f"color: {C_TEXT};")
        strength_layout.addWidget(self.lbl_strength)

        self.slider_strength = QSlider(Qt.Horizontal)
        self.slider_strength.setRange(10, 100)
        self.slider_strength.setValue(100)
        self.slider_strength.setTickInterval(10)
        self.slider_strength.valueChanged.connect(self._on_strength_changed)
        strength_layout.addWidget(self.slider_strength)

        layout.addWidget(strength_box)

        # ── Presets ──
        preset_box = make_panel("Presets")
        preset_layout = QVBoxLayout(preset_box)
        preset_layout.setSpacing(6)

        for preset_name in PRESETS.keys():
            btn = QPushButton(f"◈  {preset_name}")
            btn.clicked.connect(lambda _, p=preset_name: self._on_preset(p))
            preset_layout.addWidget(btn)

        layout.addWidget(preset_box)

        # ── Recording ──
        rec_box = make_panel("Recording")
        rec_layout = QVBoxLayout(rec_box)
        rec_layout.setSpacing(8)

        self.btn_record = QPushButton("⏺  Start Recording")
        self.btn_record.setObjectName("record")
        self.btn_record.setEnabled(False)
        self.btn_record.clicked.connect(self._on_toggle_recording)
        rec_layout.addWidget(self.btn_record)

        self.btn_screenshot = QPushButton("📸  Screenshot")
        self.btn_screenshot.setEnabled(False)
        self.btn_screenshot.clicked.connect(self._on_screenshot)
        rec_layout.addWidget(self.btn_screenshot)

        layout.addWidget(rec_box)

        # ── Detector-Info ──
        det_box = make_panel("Detector")
        det_layout = QVBoxLayout(det_box)

        self.lbl_detector = QLabel(f"Engine: {self.detector.detector_name}")
        self.lbl_detector.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 11px;")
        self.lbl_detector.setWordWrap(True)
        det_layout.addWidget(self.lbl_detector)

        layout.addWidget(det_box)

        # ── Debug Overlay Toggle ──
        dbg_box = make_panel("Options")
        dbg_layout = QVBoxLayout(dbg_box)

        self.btn_overlay = QPushButton("⊕  Show Debug Info")
        self.btn_overlay.setCheckable(True)
        self.btn_overlay.setChecked(True)
        self.btn_overlay.clicked.connect(self._on_toggle_overlay)
        dbg_layout.addWidget(self.btn_overlay)

        layout.addWidget(dbg_box)

        layout.addStretch()
        return panel

    # ── Status Bar ───────────────────────────────────────────────────────────

    def _setup_status_bar(self):
        sb = QStatusBar()
        self.setStatusBar(sb)
        self.lbl_status = QLabel("Ready  ·  FaceCensor Pro v2.0")
        sb.addWidget(self.lbl_status)

    # ── Kamera-Steuerung ─────────────────────────────────────────────────────

    def _on_start_camera(self):
        """Kamera starten."""
        self.lbl_status.setText("Starting camera...")

        if not self.camera.start():
            self.lbl_status.setText("❌ Camera not found! Check connection.")
            return

        # Processing-Thread starten
        self.proc_thread = ProcessingThread(self.camera, self.detector)
        self.proc_thread.effect_name = self._active_effect
        self.proc_thread.strength = self.slider_strength.value() / 100.0
        self.proc_thread.frame_ready.connect(self._on_frame_ready)
        self.proc_thread.start()

        # UI-Zustand anpassen
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_record.setEnabled(True)
        self.btn_screenshot.setEnabled(True)
        self.lbl_status.setText(
            f"Camera running  ·  Detector: {self.detector.detector_name}"
        )
        self.lbl_rec.setText("Live")

    def _on_stop_camera(self):
        """Kamera und Thread stoppen."""
        if self.proc_thread:
            self.proc_thread.stop()
            self.proc_thread = None

        # Aufnahme stoppen falls aktiv
        if self.recorder.is_recording:
            self.recorder.stop()

        self.camera.stop()

        # UI zurücksetzen
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_record.setEnabled(False)
        self.btn_screenshot.setEnabled(False)
        self.video_label.setText("Camera stopped\n\nPress  ▶ Start Camera  to begin")
        self.lbl_fps.setText("–")
        self.lbl_faces.setText("–")
        self.lbl_rec.setText("–")
        self.lbl_status.setText("Camera stopped")

    # ── Frame verarbeiten und anzeigen ────────────────────────────────────────

    def _on_frame_ready(self, frame: np.ndarray, face_count: int, fps: float):
        """
        Slot: Empfängt verarbeiteten Frame vom Processing-Thread.
        Konvertiert BGR → RGB → QImage → QPixmap für Anzeige.
        """
        # Recording
        if self.recorder.is_recording:
            self.recorder.write_frame(frame)

        # Statistiken aktualisieren
        self.lbl_fps.setText(f"{fps:.0f}")
        self.lbl_faces.setText(str(face_count))
        self.lbl_mode.setText(
            self._active_effect.replace(" ", "\n") if len(self._active_effect) > 10
            else self._active_effect
        )

        # Frame für Anzeige konvertieren
        display = frame.copy()

        # Debug-Overlay
        if self.proc_thread and self.proc_thread.show_overlay:
            self._draw_overlay(display, face_count, fps)

        # BGR → RGB (OpenCV nutzt BGR, Qt erwartet RGB)
        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # Auf Label-Größe skalieren (Aspect-Ratio erhalten)
        pixmap = QPixmap.fromImage(qt_image)
        scaled = pixmap.scaled(
            self.video_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.video_label.setPixmap(scaled)

    def _draw_overlay(self, frame: np.ndarray, face_count: int, fps: float):
        """Zeichnet Debug-Info ins Frame."""
        h, w = frame.shape[:2]
        # Leicht transparentes Hintergrund-Rechteck
        overlay = frame.copy()
        cv2.rectangle(overlay, (8, 8), (300, 70), (31, 42, 54), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f"FPS: {fps:.1f}  |  Faces: {face_count}", (16, 32),
                    font, 0.55, (203, 204, 190), 1, cv2.LINE_AA)
        effect_text = f"Effect: {self._active_effect}"
        if self.proc_thread and self.proc_thread.effect_name == "emoji":
            emoji_name = self.proc_thread.emoji_name or ""
            effect_text = f"Effect: Emoji  [{emoji_name}]"
        cv2.putText(frame, effect_text, (16, 55),
                    font, 0.45, (138, 154, 168), 1, cv2.LINE_AA)

    # ── Effekt-Buttons ───────────────────────────────────────────────────────

    def _on_select_effect(self, effect_key: str):
        """Effekt auswählen."""
        self._active_effect = effect_key
        self._update_effect_buttons()
        if self.proc_thread:
            self.proc_thread.effect_name = effect_key
        # Emoji-Modus deaktivieren
        self.btn_emoji_toggle.setChecked(False)
        self.emoji_combo.setEnabled(False)
        self.lbl_status.setText(f"Effect: {effect_key}")

    def _update_effect_buttons(self):
        """Aktualisiert den visuellen Zustand der Effekt-Buttons."""
        for key, btn in self._effect_buttons.items():
            if key == self._active_effect:
                btn.setObjectName("active")
                btn.setChecked(True)
            else:
                btn.setObjectName("")
                btn.setChecked(False)
            # Style neu anwenden (Qt-Quirk)
            btn.style().unpolish(btn)
            btn.style().polish(btn)

    # ── Emoji ────────────────────────────────────────────────────────────────

    def _on_toggle_emoji(self, checked: bool):
        """Emoji-Modus ein-/ausschalten."""
        if checked:
            self._active_effect = "emoji"
            self.emoji_combo.setEnabled(True)
            emoji_name = self.emoji_combo.currentText()
            if self.proc_thread:
                self.proc_thread.effect_name = "emoji"
                self.proc_thread.emoji_name = emoji_name
            # Alle Effekt-Buttons deaktivieren
            for btn in self._effect_buttons.values():
                btn.setChecked(False)
                btn.setObjectName("")
                btn.style().unpolish(btn)
                btn.style().polish(btn)
            self.lbl_status.setText(f"Emoji Mode: {emoji_name}")
        else:
            self.emoji_combo.setEnabled(False)
            self._active_effect = "Blur Gaussian"
            self._update_effect_buttons()
            if self.proc_thread:
                self.proc_thread.effect_name = "Blur Gaussian"
                self.proc_thread.emoji_name = None

    def _on_emoji_changed(self, name: str):
        """Emoji-Auswahl geändert."""
        if self.proc_thread:
            self.proc_thread.emoji_name = name
        self.lbl_status.setText(f"Emoji: {name}")

    # ── Stärke-Slider ────────────────────────────────────────────────────────

    def _on_strength_changed(self, value: int):
        """Effekt-Stärke anpassen."""
        strength = value / 100.0
        self.lbl_strength.setText(f"Intensity:  {value}%")
        if self.proc_thread:
            self.proc_thread.strength = strength

    # ── Presets ──────────────────────────────────────────────────────────────

    def _on_preset(self, preset_name: str):
        """Preset anwenden."""
        preset = PRESETS.get(preset_name)
        if not preset:
            return

        effect = preset["effect"]
        strength = preset.get("strength", 1.0)

        # Stärke setzen
        self.slider_strength.setValue(int(strength * 100))

        if effect == "emoji":
            emoji_name = preset.get("emoji", self.emoji_combo.currentText())
            self.emoji_combo.setCurrentText(emoji_name)
            self.btn_emoji_toggle.setChecked(True)
            self._on_toggle_emoji(True)
        else:
            self.btn_emoji_toggle.setChecked(False)
            self._on_toggle_emoji(False)
            self._on_select_effect(effect)

        self.lbl_status.setText(f"Preset applied: {preset_name}")

    # ── Recording ────────────────────────────────────────────────────────────

    def _on_toggle_recording(self):
        """Aufnahme starten / stoppen."""
        if self.recorder.is_recording:
            filename = self.recorder.stop()
            self.btn_record.setText("⏺  Start Recording")
            self.lbl_rec.setText("Live")
            self.lbl_status.setText(f"Recording saved: {filename}")
        else:
            # Frame-Größe ermitteln
            filename = self.recorder.start(width=640, height=480, fps=25.0)
            if filename:
                self.btn_record.setText("⏹  Stop Recording")
                self.lbl_rec.setText("● REC")
                self.lbl_status.setText(f"Recording: {filename}")
            else:
                self.lbl_status.setText("❌ Recording failed!")

    # ── Screenshot ───────────────────────────────────────────────────────────

    def _on_screenshot(self):
        """Screenshot des aktuellen Frames speichern."""
        # Letzten Frame aus Queue holen
        ret, frame = self.camera.read()
        if ret and frame is not None:
            # Effekt für Screenshot auch anwenden
            faces = self.detector.detect(frame)
            effect = self._active_effect
            emoji = self.proc_thread.emoji_name if self.proc_thread else None
            strength = self.slider_strength.value() / 100.0
            for (x, y, w, h) in faces:
                frame = apply_effect(frame, x, y, w, h, effect, strength, emoji)
            filename = save_screenshot(frame, output_dir="screenshots")
            self.lbl_status.setText(f"Screenshot saved: {filename}")
        else:
            self.lbl_status.setText("❌ No frame available for screenshot")

    # ── Debug Overlay Toggle ─────────────────────────────────────────────────

    def _on_toggle_overlay(self, checked: bool):
        """Debug-Overlay ein-/ausschalten."""
        if self.proc_thread:
            self.proc_thread.show_overlay = checked

    # ── Keyboard Shortcuts ───────────────────────────────────────────────────

    def keyPressEvent(self, event):
        """Keyboard-Shortcuts (zusätzlich zu den Buttons)."""
        key = event.key()
        if key == Qt.Key_Escape:
            self.close()
        elif key == Qt.Key_Space:
            if self.camera.is_running():
                self._on_stop_camera()
            else:
                self._on_start_camera()
        elif key == Qt.Key_S:
            if self.btn_screenshot.isEnabled():
                self._on_screenshot()
        elif key == Qt.Key_R:
            if self.btn_record.isEnabled():
                self._on_toggle_recording()

    # ── Sauberes Beenden ─────────────────────────────────────────────────────

    def closeEvent(self, event):
        """Sauberes Herunterfahren: Thread + Kamera + Recording freigeben."""
        logger.info("Anwendung wird beendet...")

        # Recording stoppen
        if self.recorder.is_recording:
            self.recorder.stop()

        # Processing-Thread stoppen
        if self.proc_thread:
            self.proc_thread.stop()

        # Kamera freigeben
        self.camera.stop()

        event.accept()
        logger.info("✅ Sauber beendet")
