#!/usr/bin/python3
"""
FaceCensor Pro - Jetson Nano Edition
=====================================
Professionelles Creator-Tool für automatische Gesichts-Anonymisierung.
Einstiegspunkt der Applikation.
"""

import sys
import os

# Stelle sicher, dass das Verzeichnis im Pfad ist
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from ui import MainWindow


def main():
    # High-DPI Unterstützung (optional, für bessere Skalierung)
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setApplicationName("FaceCensor Pro")
    app.setApplicationVersion("2.0")

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
