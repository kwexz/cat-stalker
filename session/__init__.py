"""Autonomous data-collection sessions.

Phone (P30) streams lightweight telemetry + event frames over Wi-Fi.
PC persists everything, keeps the Dreame Wi-Fi awake, watches
watchdogs and signals the human via the robot speaker.

The Windows/Python prototype `cat_stalker.py` is intentionally NOT
imported here (it pulls pygame/cv2/YOLO at module import time).
The tiny miIO wrapper below duplicates only the proven Dreame mapping.
"""
