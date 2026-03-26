#!/usr/bin/python3
"""
recorder.py - Video-Aufnahme Modul
====================================
Schreibt den verarbeiteten (zensierten) Video-Stream in eine Datei.
Nutzt OpenCV VideoWriter mit MP4V-Codec.
"""

import cv2
import os
import threading
import queue
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


class VideoRecorder:
    """
    Asynchroner Video-Recorder.
    
    Schreibt Frames in eigenem Thread, damit Recording die UI nicht blockiert.
    """

    def __init__(self, output_dir: str = "."):
        self.output_dir = output_dir
        self._writer: Optional[cv2.VideoWriter] = None
        self._recording = False
        self._frame_queue: queue.Queue = queue.Queue(maxsize=30)
        self._thread: Optional[threading.Thread] = None
        self._current_file: Optional[str] = None
        os.makedirs(output_dir, exist_ok=True)

    def start(self, width: int, height: int, fps: float = 30.0) -> str:
        """Startet die Aufnahme. Gibt den Dateinamen zurück."""
        if self._recording:
            self.stop()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.output_dir, f"facecensor_{timestamp}.mp4")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self._writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))

        if not self._writer.isOpened():
            logger.error(f"❌ VideoWriter konnte nicht geöffnet werden: {filename}")
            self._writer = None
            return ""

        self._current_file = filename
        self._recording = True

        self._thread = threading.Thread(
            target=self._write_loop,
            daemon=True,
            name="VideoRecorder"
        )
        self._thread.start()
        logger.info(f"🔴 Aufnahme gestartet: {filename}")
        return filename

    def _write_loop(self):
        """Schreibt Frames aus der Queue in die Datei."""
        while self._recording or not self._frame_queue.empty():
            try:
                frame = self._frame_queue.get(timeout=0.5)
                if self._writer and frame is not None:
                    self._writer.write(frame)
            except queue.Empty:
                continue

    def write_frame(self, frame):
        """Frame zur Aufnahme hinzufügen (non-blocking)."""
        if not self._recording:
            return
        try:
            self._frame_queue.put_nowait(frame.copy())
        except queue.Full:
            pass  # Frame verwerfen wenn Queue voll

    def stop(self) -> Optional[str]:
        """Stoppt die Aufnahme und gibt den Dateinamen zurück."""
        if not self._recording:
            return None

        self._recording = False

        if self._thread:
            self._thread.join(timeout=5.0)

        if self._writer:
            self._writer.release()
            self._writer = None

        filename = self._current_file
        self._current_file = None
        logger.info(f"⏹ Aufnahme beendet: {filename}")
        return filename

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def current_file(self) -> Optional[str]:
        return self._current_file


def save_screenshot(frame, output_dir: str = ".") -> str:
    """Speichert einen einzelnen Frame als JPEG."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f"screenshot_{timestamp}.jpg")
    cv2.imwrite(filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    logger.info(f"📸 Screenshot: {filename}")
    return filename
