import os

import cv2
import numpy as np
import pytest

from calibration import (DEFAULT_PX_PER_CM, ScaleCalibration, describe, detect_marker, load_calibration,
                         save_calibration, save_marker_pdf)
from conftest import PX_PER_CM, make_scene


def test_detects_marker_scale():
    cal = detect_marker(make_scene([]), 5.0)
    assert cal is not None
    assert cal.px_per_cm == pytest.approx(PX_PER_CM, rel=0.02)


def test_marker_size_changes_scale():
    assert detect_marker(make_scene([]), 10.0).px_per_cm == pytest.approx(PX_PER_CM / 2, rel=0.02)


def test_to_cm_maps_marker_corners():
    cal = detect_marker(make_scene([]), 5.0)
    assert np.allclose(cal.to_cm(cal.corners), [[0, 0], [5, 0], [5, 5], [0, 5]], atol=1e-4)


def test_no_marker_returns_none():
    assert detect_marker(make_scene([], marker_cm=0), 5.0) is None


def test_invalid_marker_size():
    with pytest.raises(ValueError):
        detect_marker(make_scene([]), 0)


def test_grayscale_input():
    gray = cv2.cvtColor(make_scene([]), cv2.COLOR_BGR2GRAY)
    assert detect_marker(gray, 5.0) is not None


def test_save_and_load_calibration(tmp_path):
    cal = detect_marker(make_scene([]), 5.0)
    path = tmp_path / 'cal.json'
    save_calibration(str(path), cal)
    loaded = load_calibration(str(path))
    assert loaded.source == 'tersimpan'
    assert loaded.image_size == (1280, 720)
    assert np.allclose(loaded.homography, cal.homography)
    assert loaded.px_per_cm == pytest.approx(cal.px_per_cm)


def test_load_missing_calibration_returns_none(tmp_path):
    assert load_calibration(str(tmp_path / 'tidak_ada.json')) is None


@pytest.mark.parametrize('content', ['{rusak', '{}', '{"homography": [1], "corners": [], '
                                     '"marker_size_cm": 5, "image_size": [1, 1]}'])
def test_load_invalid_calibration_raises(tmp_path, content):
    path = tmp_path / 'cal.json'
    path.write_text(content)
    with pytest.raises(ValueError):
        load_calibration(str(path))


def test_calibration_for_frame_rescales_same_aspect():
    cal = detect_marker(make_scene([]), 5.0)
    half = cal.for_frame(640, 360)
    assert half.px_per_cm == pytest.approx(cal.px_per_cm / 2, rel=1e-6)
    # titik yang sama secara fisik harus memetakan ke cm yang sama
    assert np.allclose(half.to_cm([[320, 180]]), cal.to_cm([[640, 360]]))
    assert cal.for_frame(1280, 720) is cal
    assert cal.for_frame(720, 720) is None


def test_scale_calibration():
    cal = ScaleCalibration(DEFAULT_PX_PER_CM)
    assert cal.to_cm([[DEFAULT_PX_PER_CM * 2, 0]]).tolist() == [[2.0, 0.0]]
    assert cal.source == 'default'
    assert 'perkiraan' in describe(cal)


def test_save_marker_pdf(tmp_path):
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from PyQt5.QtWidgets import QApplication
    _app = QApplication.instance() or QApplication([])
    out = tmp_path / 'marker.pdf'
    save_marker_pdf(str(out), 5.0)
    assert out.read_bytes()[:5] == b'%PDF-'
