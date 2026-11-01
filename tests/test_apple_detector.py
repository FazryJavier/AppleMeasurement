import cv2
import numpy as np
import pytest

from AppleDetector import (NOT_CALIBRATED, AppleDetector, DiameterSmoother, grade_apple,
                           measure_apples)
from calibration import detect_marker
from conftest import PX_PER_CM, make_scene

RED = (30, 30, 200)
GREEN = (60, 200, 140)
YELLOW = (40, 200, 230)


def measure(img, marker_cm=5.0):
    return measure_apples(img, detect_marker(img, marker_cm))


@pytest.mark.parametrize('diameter, expected', [
    (8.0, (12200, 'PREMIUM')),
    (7.0, (12200, 'PREMIUM')),
    (6.9, (9500, 'REGULER')),
    (6.95, (9500, 'REGULER')),
    (6.0, (9500, 'REGULER')),
    (5.9, (7500, 'SUPER SAVER')),
    (5.0, (7500, 'SUPER SAVER')),
    (4.9, (0, 'TERLALU KECIL')),
    (8.1, (None, 'TIDAK VALID')),
])
def test_grade_apple(diameter, expected):
    assert grade_apple(diameter) == expected


@pytest.mark.parametrize('color, name', [(RED, 'MERAH'), (GREEN, 'HIJAU'), (YELLOW, 'KUNING')])
def test_measures_apples_of_any_color(color, name):
    _, results = measure(make_scene([(16, 9, 7.5, color)]))
    assert len(results) == 1
    assert results[0]['diameter'] == pytest.approx(7.5, abs=0.1)
    assert results[0]['grade'] == 'PREMIUM'
    assert results[0]['color'] == name


def test_measures_every_apple_and_keeps_input_intact():
    img = make_scene([(12, 9, 7.5, RED), (22, 9, 5.5, GREEN)])
    original = img.copy()
    annotated, results = measure(img)
    by_grade = {r['grade']: r['diameter'] for r in results}
    assert by_grade['PREMIUM'] == pytest.approx(7.5, abs=0.1)
    assert by_grade['SUPER SAVER'] == pytest.approx(5.5, abs=0.1)
    assert np.array_equal(img, original)
    assert not np.array_equal(annotated, original)


def test_respects_marker_size():
    # marker yang sama diklaim 10 cm -> semua ukuran jadi 2x
    _, results = measure(make_scene([(16, 9, 3.5, RED)]), marker_cm=10.0)
    assert results[0]['diameter'] == pytest.approx(7.0, abs=0.1)


def test_perspective_is_corrected():
    """Foto diambil miring: homografi marker harus mengoreksi distorsi."""
    img = make_scene([(16, 9, 6.5, RED)])
    h, w = img.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[80, 40], [w - 30, 0], [w, h], [0, h - 60]])
    warped = cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (w, h),
                                 borderValue=(235, 235, 235))
    _, results = measure(warped)
    assert len(results) == 1
    assert results[0]['diameter'] == pytest.approx(6.5, abs=0.15)


def test_without_marker_no_price():
    _, results = measure(make_scene([(16, 9, 7.5, RED)], marker_cm=0))
    assert results[0]['diameter'] is None
    assert results[0]['price'] is None
    assert results[0]['grade'] == NOT_CALIBRATED


def test_no_apple_does_not_crash():
    _, results = measure(make_scene([]))
    assert results == []


def test_stem_and_highlight_do_not_change_diameter():
    img = make_scene([(16, 9, 7.0, RED)])
    cx, cy, r = 16 * PX_PER_CM, 9 * PX_PER_CM, int(3.5 * PX_PER_CM)
    cv2.line(img, (cx, cy - r + 10), (cx + 15, cy - r - 50), (30, 50, 170), 8)  # tangkai kemerahan
    cv2.circle(img, (cx - 40, cy - 40), 25, (240, 240, 250), -1)  # pantulan cahaya
    cv2.circle(img, (cx + r - 5, cy), 18, (230, 230, 240), -1)  # pantulan di tepi
    _, results = measure(img)
    assert len(results) == 1
    assert results[0]['diameter'] == pytest.approx(7.0, abs=0.15)


def test_rejects_non_apple_objects():
    img = make_scene([])
    cv2.rectangle(img, (500, 200), (900, 300), RED, -1)  # benda merah panjang
    cv2.circle(img, (300, 500), 120, (150, 170, 210), -1)  # warna kulit / kayu pucat
    cv2.circle(img, (1000, 500), 10, RED, -1)  # noise kecil
    _, results = measure(img)
    assert results == []


def test_rejects_apple_cut_by_frame_border():
    detector = AppleDetector()
    img = make_scene([(1, 9, 7.0, RED)])
    _, results = measure_apples(img, detect_marker(img, 5.0), detector)
    assert results == []
    assert detector.last_rejected['terpotong tepi'] == 1


def test_apple_barely_touching_border_is_measured_with_warning():
    # tepi bawah frame = 18 cm; apel 7 cm berpusat di 14.5 cm menyentuh tepi
    _, results = measure(make_scene([(16, 14.5, 7.0, RED)]))
    assert len(results) == 1
    assert results[0]['at_border']
    assert results[0]['diameter'] == pytest.approx(7.0, abs=0.15)


def test_attached_leaf_is_ignored():
    img = make_scene([(16, 10, 7.0, RED)])
    cx, top = 16 * PX_PER_CM, int((10 - 3.5) * PX_PER_CM)
    # daun tersambung ke apel lewat tangkai, seperti pada foto apel umumnya
    cv2.line(img, (cx, top + 10), (cx + 40, top - 40), (40, 140, 60), 6)
    cv2.ellipse(img, (cx + 100, top - 60), (65, 32), -20, 0, 360, (40, 140, 60), -1)
    detector = AppleDetector()
    _, results = measure_apples(img, detect_marker(img, 5.0), detector)
    assert [r['color'] for r in results] == ['MERAH']
    assert results[0]['diameter'] == pytest.approx(7.0, abs=0.15)


def test_separate_leaf_next_to_apple_is_ignored():
    # daun jadi blob sendiri yang nyaris menempel apel (seperti img/Apel 8.jpg)
    img = make_scene([(16, 10, 7.0, RED)])
    cx, cy, r = 16 * PX_PER_CM, 10 * PX_PER_CM, int(3.5 * PX_PER_CM)
    leaf_center = (cx + int(r * 0.71) + 65, cy - int(r * 0.71) - 65)  # celah ~20 px
    cv2.circle(img, leaf_center, 70, (40, 140, 60), -1)
    detector = AppleDetector()
    _, results = measure_apples(img, detect_marker(img, 5.0), detector)
    assert [r['color'] for r in results] == ['MERAH']
    assert detector.last_rejected['kemungkinan daun'] == 1


def test_separate_small_green_apple_is_not_a_leaf():
    _, results = measure(make_scene([(10, 9, 8.0, RED), (22, 9, 4.0, GREEN)]))
    assert sorted(r['color'] for r in results) == ['HIJAU', 'MERAH']


def test_apple_touching_object_of_other_color_is_split():
    img = make_scene([(14, 9, 6.5, RED)])
    # benda kuning panjang (mis. pisang) menempel di sisi kanan apel
    cv2.rectangle(img, (14 * PX_PER_CM + 120, 200), (14 * PX_PER_CM + 500, 300), YELLOW, -1)
    _, results = measure(img)
    assert len(results) == 1
    assert results[0]['color'] == 'MERAH'
    assert results[0]['diameter'] == pytest.approx(6.5, abs=0.15)


def test_rejection_reason_for_non_round_object():
    detector = AppleDetector()
    img = make_scene([])
    cv2.rectangle(img, (500, 200), (900, 300), RED, -1)
    measure_apples(img, None, detector)
    assert detector.last_rejected['tidak bulat'] == 1


def test_last_mask_matches_detection():
    detector = AppleDetector()
    img = make_scene([(16, 9, 7.0, RED)])
    measure_apples(img, None, detector)
    assert detector.last_mask.shape == img.shape[:2]
    assert cv2.countNonZero(detector.last_mask) > 0


def result(diameter, center=(100, 100)):
    return {'diameter': diameter, 'price': None, 'grade': '', 'color': 'MERAH', 'center': center}


def test_smoother_uses_median_per_apple():
    s = DiameterSmoother(window=5)
    for d in (7.0, 7.1, 9.9, 6.9, 7.0):  # 9.9 = outlier sesaat
        out = s.update([result(d)], (640, 480))
    assert out[0]['diameter'] == 7.0
    assert out[0]['grade'] == 'PREMIUM'


def test_smoother_tracks_apples_separately():
    s = DiameterSmoother()
    for _ in range(3):
        out = s.update([result(7.5, (100, 100)), result(5.5, (500, 300))], (640, 480))
    assert [r['diameter'] for r in out] == [7.5, 5.5]


def test_smoother_drops_lost_track():
    s = DiameterSmoother(max_missed=0)
    s.update([result(7.5, (100, 100))], (640, 480))
    s.update([], (640, 480))
    out = s.update([result(5.5, (100, 100))], (640, 480))
    assert out[0]['diameter'] == 5.5


def test_smoother_keeps_uncalibrated_results():
    out = DiameterSmoother().update([result(None)], (640, 480))
    assert out[0]['diameter'] is None
