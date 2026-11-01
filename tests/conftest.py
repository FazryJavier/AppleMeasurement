import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from calibration import marker_image  # noqa: E402

PX_PER_CM = 40  # skala adegan sintetis


def make_scene(apples, marker_cm=5.0, marker_pos=(40, 40), size=(720, 1280), background=(235, 235, 235)):
    """Adegan sintetis: latar polos + marker ArUco + apel (lingkaran).

    apples: list of (pusat_x_cm, pusat_y_cm, diameter_cm, warna_bgr)."""
    img = np.full((size[0], size[1], 3), background, dtype=np.uint8)
    if marker_cm:
        side = int(round(marker_cm * PX_PER_CM))
        m = cv2.cvtColor(marker_image(side), cv2.COLOR_GRAY2BGR)
        x, y = marker_pos
        img[y:y + side, x:x + side] = m
    for cx, cy, d, color in apples:
        center = (int(round(cx * PX_PER_CM)), int(round(cy * PX_PER_CM)))
        cv2.circle(img, center, int(round(d * PX_PER_CM / 2)), color, -1, cv2.LINE_AA)
    return img
