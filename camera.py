#!/usr/bin/python3
"""
camera.py - Kamera-Handling mit dediziertem Capture-Thread
===========================================================
PERFORMANCE-OPTIMIERUNG:
- Separater Thread für Kamera-Capture, damit UI nie blockiert
- Queue mit max. 2 Frames verhindert Aufstauen alter Frames
- Direkte GStreamer-Pipeline für Jetson Nano (Hardware-Decoder)
- Falls CSI nicht verfügbar: automatischer Fallback auf USB/V4L2
"""

import cv2
import threading
import queue
import time
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def gstreamer_csi_pipeline(
    sensor_id: int = 0,
    capture_width: int = 1280,
    capture_height: int = 720,
    display_width: int = 640,
    display_height: int = 480,
    framerate: int = 30,
    flip_method: int = 0,
) -> str:
    """
    Erzeugt GStreamer-Pipeline für Jetson Nano CSI-Kamera.
    Nutzt nvarguscamerasrc + nvvidconv (Hardware-beschleunigt).
    """
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width={capture_width}, height={capture_height}, "
        f"framerate={framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width={display_width}, height={display_height}, format=BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=BGR ! appsink drop=1"
        # drop=1: Frames werden verworfen wenn appsink voll → kein Anstauen!
    )


def gstreamer_usb_pipeline(
    device: int = 0,
    width: int = 640,
    height: int = 480,
    framerate: int = 30,
) -> str:
    """
    GStreamer-Pipeline für USB-Kamera (V4L2).
    Auch auf Jetson Nano nutzbar für USB-Webcams.
    """
    return (
        f"v4l2src device=/dev/video{device} ! "
        f"video/x-raw, width={width}, height={height}, framerate={framerate}/1 ! "
        f"videoconvert ! "
        f"video/x-raw, format=BGR ! appsink drop=1"
    )


class CameraCapture:
    """
    Thread-basierter Kamera-Capture.
    
    PERFORMANCE:
    - Läuft in eigenem Thread → UI und Processing blockieren sich nicht
    - Queue(maxsize=2): Immer nur die aktuellsten Frames werden gehalten
    - Bei vollem Queue wird der älteste Frame verworfen (non-blocking put)
    """

    def __init__(
        self,
        sensor_id: int = 0,
        capture_width: int = 1280,
        capture_height: int = 720,
        display_width: int = 640,
        display_height: int = 480,
        framerate: int = 30,
        flip_method: int = 0,
    ):
        self.sensor_id = sensor_id
        self.capture_width = capture_width
        self.capture_height = capture_height
        self.display_width = display_width
        self.display_height = display_height
        self.framerate = framerate
        self.flip_method = flip_method

        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_queue: queue.Queue = queue.Queue(maxsize=2)
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # FPS-Tracking
        self._fps = 0.0
        self._frame_count = 0
        self._fps_timer = time.time()

    def open(self) -> bool:
        """
        Öffnet die Kamera. Versucht in Reihenfolge:
        1. CSI-Kamera via GStreamer (Jetson Nano nativ)
        2. USB-Kamera via GStreamer
        3. OpenCV-Fallback (Standard-V4L2)
        """
        # Versuch 1: CSI Kamera (Jetson Nano)
        pipeline = gstreamer_csi_pipeline(
            sensor_id=self.sensor_id,
            capture_width=self.capture_width,
            capture_height=self.capture_height,
            display_width=self.display_width,
            display_height=self.display_height,
            framerate=self.framerate,
            flip_method=self.flip_method,
        )
        logger.info("Versuche CSI-Kamera über GStreamer...")
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            self.cap = cap
            logger.info("✅ CSI-Kamera geöffnet (GStreamer)")
            return True

        # Versuch 2: USB via GStreamer
        logger.info("CSI fehlgeschlagen. Versuche USB über GStreamer...")
        usb_pipe = gstreamer_usb_pipeline(
            device=self.sensor_id,
            width=self.display_width,
            height=self.display_height,
            framerate=self.framerate,
        )
        cap = cv2.VideoCapture(usb_pipe, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            self.cap = cap
            logger.info("✅ USB-Kamera geöffnet (GStreamer)")
            return True

        # Versuch 3: OpenCV-Fallback
        logger.info("GStreamer fehlgeschlagen. Versuche OpenCV-Fallback...")
        cap = cv2.VideoCapture(self.sensor_id)
        if cap.isOpened():
            # Kamera-Settings optimieren
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.display_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.display_height)
            cap.set(cv2.CAP_PROP_FPS, self.framerate)
            # Buffer auf 1 setzen → minimale Latenz
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap = cap
            logger.info("✅ Kamera geöffnet (OpenCV-Fallback)")
            return True

        logger.error("❌ Keine Kamera gefunden!")
        return False

    def start(self) -> bool:
        """Startet den Capture-Thread."""
        if not self.open():
            return False
        self._running = True
        self._thread = threading.Thread(
            target=self._capture_loop,
            daemon=True,  # Thread beendet sich mit Hauptprogramm
            name="CameraCapture"
        )
        self._thread.start()
        logger.info("📹 Capture-Thread gestartet")
        return True

    def _capture_loop(self):
        """
        Capture-Schleife im eigenen Thread.
        PERFORMANCE: Liest kontinuierlich Frames und legt nur den neuesten in die Queue.
        """
        while self._running:
            if self.cap is None or not self.cap.isOpened():
                break

            ret, frame = self.cap.read()
            if not ret or frame is None:
                logger.warning("Frame konnte nicht gelesen werden")
                time.sleep(0.01)
                continue

            # Queue befüllen: alten Frame verwerfen wenn voll
            # → immer nur aktuellster Frame verfügbar (kein Aufstauen)
            try:
                self.frame_queue.put_nowait(frame)
            except queue.Full:
                try:
                    self.frame_queue.get_nowait()  # Alten Frame raus
                    self.frame_queue.put_nowait(frame)
                except (queue.Empty, queue.Full):
                    pass

            # FPS berechnen (alle 30 Frames)
            self._frame_count += 1
            if self._frame_count % 30 == 0:
                elapsed = time.time() - self._fps_timer
                self._fps = 30.0 / elapsed if elapsed > 0 else 0.0
                self._fps_timer = time.time()

    def read(self) -> Tuple[bool, Optional[any]]:
        """Liest den neuesten verfügbaren Frame."""
        try:
            frame = self.frame_queue.get(timeout=0.1)
            return True, frame
        except queue.Empty:
            return False, None

    @property
    def fps(self) -> float:
        return self._fps

    def stop(self):
        """Stoppt den Capture-Thread und gibt Ressourcen frei."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if self.cap:
            self.cap.release()
            self.cap = None
        logger.info("📹 Kamera freigegeben")

    def is_running(self) -> bool:
        return self._running and self.cap is not None
