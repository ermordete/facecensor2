# FaceCensor Pro – Jetson Nano Edition

Professionelles Creator-Tool für automatische Echtzeit-Gesichts-Anonymisierung.

---

## Installation

### 1. System-Pakete (Jetson Nano / Ubuntu)

```bash
sudo apt update
sudo apt install -y \
    python3-pip \
    python3-pyqt5 \
    libopencv-dev \
    python3-opencv \
    libgstreamer1.0-dev \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad
```

### 2. Python-Pakete

```bash
pip3 install numpy
# Falls PyQt5 nicht via apt verfügbar:
pip3 install PyQt5
```

> **Hinweis Jetson Nano:** OpenCV ist auf dem Nano meist vorinstalliert (mit GStreamer-Unterstützung). Falls nicht: `pip3 install opencv-python` — dies enthält aber keinen GStreamer-Support. Besser: OpenCV aus den JetPack-Paketen nutzen.

---

## DNN Face Detector (empfohlen, deutlich besser als Haar Cascade)

Das SSD-Modell muss einmalig heruntergeladen werden:

```bash
mkdir -p models
cd models

# Prototxt
wget https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt

# Caffemodel
wget https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel
```

Falls kein Download möglich: App fällt automatisch auf Haar Cascade zurück.

---

## Starten

```bash
cd facecensor
python3 main.py
```

### Keyboard Shortcuts

| Taste   | Funktion          |
|---------|-------------------|
| `Space` | Start / Stop Cam  |
| `S`     | Screenshot        |
| `R`     | Recording Toggle  |
| `ESC`   | Beenden           |

---

## Verzeichnisstruktur

```
facecensor/
├── main.py          # Einstiegspunkt
├── ui.py            # PyQt5 GUI
├── camera.py        # Kamera-Capture (Thread-basiert)
├── detector.py      # Gesichtserkennung (DNN + Haar Fallback)
├── effects.py       # Alle Zensur-Effekte
├── recorder.py      # Video-Aufnahme
├── models/          # DNN-Modell-Dateien (s. Download oben)
│   ├── deploy.prototxt
│   └── res10_300x300_ssd_iter_140000.caffemodel
├── emojis/          # Optionale Emoji-PNGs (RGBA)
│   ├── cool.png
│   ├── lol.png
│   └── ...
├── recordings/      # Aufnahmen (wird automatisch erstellt)
└── screenshots/     # Screenshots (wird automatisch erstellt)
```

---

## Eigene Emojis/Sticker hinzufügen

1. PNG mit Alpha-Kanal (RGBA) erstellen / herunterladen
2. In `emojis/` ablegen
3. In `effects.py` → `EmojiOverlay.AVAILABLE_EMOJIS` eintragen:
   ```python
   "🦊 Fox": ("fox.png", (255, 140, 0), "FOX"),
   ```
4. App neu starten

---

## Performance-Tipps für Jetson Nano

- **Safe Mode Preset**: Nutzt Black Bar (CPU-effizientester Effekt)
- **Kamera-Auflösung** in `main.py` bei `CameraCapture(...)` reduzieren (z.B. 480p)
- **`detect_every_n_frames=3`** für weniger CPU-Last bei Gesichtserkennung
- **GStreamer CSI-Pipeline** wird automatisch bevorzugt (Hardware-Decoder)

---

## Bekannte Einschränkungen

- Emoji-Overlays benötigen RGBA-PNGs; Fallback erzeugt synthetische Ersatz-Grafiken
- DNN-Modell muss manuell heruntergeladen werden (s. oben)
- Recording nutzt `mp4v` Codec; für H.264 ggf. `avc1` verwenden
