import os

import numpy as np
import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PyQt5 import QtWidgets  # noqa: E402

import main  # noqa: E402
from conftest import make_scene  # noqa: E402

IMG_DIR = os.path.join(main.BASE_DIR, 'img')
RED = (30, 30, 200)


@pytest.fixture(scope='module')
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def calibration_path(tmp_path):
    # jangan sentuh calibration.json milik pengguna
    return str(tmp_path / 'calibration.json')


@pytest.fixture
def window(app, calibration_path):
    w = main.ShowImage(calibration_path)
    yield w
    w.stopCamera()


def save_image(tmp_path, name, img):
    path = tmp_path / name
    main.cv2.imwrite(str(path), img)
    return str(path)


def save_scene(tmp_path, name, *args, **kwargs):
    return save_image(tmp_path, name, make_scene(*args, **kwargs))


class FakeCamera:
    def __init__(self, frames, opened=True):
        self.frames = list(frames)
        self.opened = opened
        self.released = False

    def isOpened(self):
        return self.opened

    def read(self):
        if not self.frames:
            return False, None
        return True, self.frames.pop(0)

    def release(self):
        self.released = True


def test_format_rupiah():
    assert main.format_rupiah(12200) == 'Rp12.200'


def test_summarize():
    assert main.summarize([]) == ('Rp0', 'TIDAK ADA APEL')
    assert main.summarize([{'price': 9500, 'grade': 'REGULER'}]) == ('Rp9.500', 'REGULER')
    assert main.summarize([{'price': None, 'grade': 'BELUM KALIBRASI'}]) == ('-', 'BELUM KALIBRASI')
    multi = [{'price': 12200, 'grade': 'PREMIUM'}, {'price': None, 'grade': 'TIDAK VALID'},
             {'price': 12200, 'grade': 'PREMIUM'}]
    assert main.summarize(multi) == ('Rp24.400', '2x PREMIUM, 1x TIDAK VALID')
    uncal = [{'price': None, 'grade': 'BELUM KALIBRASI'}] * 2
    assert main.summarize(uncal) == ('-', '2x BELUM KALIBRASI')


def test_detection_status():
    from collections import Counter
    assert main.detection_status([{'at_border': False}], Counter()) == ''
    status = main.detection_status([], Counter({'tidak bulat': 2}))
    assert status.startswith('Tidak ada apel terdeteksi.')
    assert '2 tidak bulat' in status
    assert 'tepi frame' in main.detection_status([{'at_border': True}], Counter())


def test_status_explains_empty_result(window, tmp_path):
    window.loadImage(save_scene(tmp_path, 'scene.png', [(1, 9, 7.0, RED)]))
    window.applemeasurement()
    message = window.statusBar().currentMessage()
    assert 'Tidak ada apel terdeteksi' in message
    assert 'terpotong tepi' in message


def test_buttons_without_image_do_not_crash(window, monkeypatch):
    warnings = []
    monkeypatch.setattr(main.QMessageBox, 'warning', lambda *a: warnings.append(a))
    window.applemeasurement()
    window.canny()
    window.reset()
    assert len(warnings) == 2


def test_static_image_with_marker(window, tmp_path):
    window.loadImage(save_scene(tmp_path, 'scene.png', [(16, 9, 6.5, RED)]))
    window.applemeasurement()
    assert window.label_16.text() == 'Rp9.500'
    assert window.label_18.text() == 'REGULER'
    assert window.output is not None
    window.reset()
    assert window.label_18.text() == '-'


def test_without_marker_uses_default_scale(window, tmp_path):
    assert 'skala default' in window.statusBar().currentMessage()
    # adegan 40 px/cm; skala default 37.8 px/cm -> 300 px = 7.9 cm
    window.loadImage(save_scene(tmp_path, 'scene.png', [(16, 9, 7.5, RED)], marker_cm=0))
    window.applemeasurement()
    assert window.label_18.text() == 'PREMIUM'
    assert 'skala default' in window.statusBar().currentMessage()


def test_saved_calibration_is_used_without_marker(app, window, tmp_path, calibration_path):
    window.loadImage(save_scene(tmp_path, 'a.png', [(16, 9, 6.5, RED)]))
    window.applemeasurement()
    info = []
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(main.QMessageBox, 'information', lambda *a: info.append(a))
        window.saveCalibration()
    assert info and os.path.exists(calibration_path)

    # aplikasi dibuka ulang: kalibrasi dimuat otomatis, marker tidak perlu
    reopened = main.ShowImage(calibration_path)
    assert 'Kalibrasi tersimpan dimuat' in reopened.statusBar().currentMessage()
    reopened.loadImage(save_scene(tmp_path, 'b.png', [(16, 9, 7.5, RED)], marker_cm=0))
    reopened.applemeasurement()
    assert reopened.label_18.text() == 'PREMIUM'
    assert 'Kalibrasi tersimpan 40.0 px/cm' in reopened.statusBar().currentMessage()

    # resolusi beda tapi rasio sama (kamera sama) -> kalibrasi diskalakan
    half = make_scene([(16, 9, 5.5, RED)], marker_cm=0)
    reopened.loadImage(save_image(tmp_path, 'half.png', main.cv2.resize(half, (640, 360))))
    reopened.applemeasurement()
    assert reopened.label_18.text() == 'SUPER SAVER'


def test_saved_calibration_ignored_for_other_aspect_ratio(window, tmp_path, monkeypatch):
    monkeypatch.setattr(main.QMessageBox, 'information', lambda *a: None)
    window.loadImage(save_scene(tmp_path, 'a.png', [(16, 9, 6.5, RED)]))
    window.applemeasurement()
    window.saveCalibration()
    window.loadImage(save_scene(tmp_path, 'sq.png', [(9, 9, 6.5, RED)], marker_cm=0, size=(720, 720)))
    window.applemeasurement()
    assert 'skala default' in window.statusBar().currentMessage()


def test_save_calibration_without_marker_warns(window, calibration_path, monkeypatch):
    warnings = []
    monkeypatch.setattr(main.QMessageBox, 'warning', lambda *a: warnings.append(a))
    window.saveCalibration()
    assert warnings and not os.path.exists(calibration_path)


def test_delete_saved_calibration(window, tmp_path, calibration_path, monkeypatch):
    monkeypatch.setattr(main.QMessageBox, 'information', lambda *a: None)
    window.loadImage(save_scene(tmp_path, 'a.png', [(16, 9, 6.5, RED)]))
    window.applemeasurement()
    window.saveCalibration()
    window.deleteSavedCalibration()
    assert not os.path.exists(calibration_path)
    assert window.saved_calibration is None
    window.deleteSavedCalibration()  # file sudah tidak ada: tidak error


def test_corrupt_calibration_file_falls_back_to_default(app, calibration_path):
    with open(calibration_path, 'w') as f:
        f.write('{rusak')
    w = main.ShowImage(calibration_path)
    assert w.saved_calibration is None
    assert 'diabaikan' in w.statusBar().currentMessage()


def test_marker_size_changes_measurement(window, tmp_path):
    window.loadImage(save_scene(tmp_path, 'scene.png', [(16, 9, 3.5, RED)]))
    window.applemeasurement()
    assert window.label_18.text() == 'TERLALU KECIL'
    window.markerSize.setValue(10.0)  # marker ternyata 10 cm -> apel 7 cm
    window.applemeasurement()
    assert window.label_18.text() == 'PREMIUM'


def test_live_camera_flow(window, monkeypatch):
    cam = FakeCamera([make_scene([(16, 9, 6.5, RED)]) for _ in range(5)])
    monkeypatch.setattr(main, 'open_camera', lambda *a: cam)
    window.actionCamera.setChecked(True)
    assert window.timer.isActive()
    assert window.buttonApple.text() == 'Ambil Foto'
    for _ in range(3):
        window.nextFrame()
    assert window.label_18.text() == 'REGULER'
    assert 'fps' in window.statusBar().currentMessage()

    window.applemeasurement()  # bekukan frame
    assert not window.timer.isActive()
    assert cam.released
    assert window.Image is not None
    assert window.label_18.text() == 'REGULER'
    assert window.buttonApple.text() == 'Apple Measurement'


def test_live_camera_read_failure_stops(window, monkeypatch):
    errors = []
    monkeypatch.setattr(main.QMessageBox, 'critical', lambda *a: errors.append(a))
    monkeypatch.setattr(main, 'open_camera', lambda *a: FakeCamera([]))
    window.actionCamera.setChecked(True)
    window.nextFrame()
    assert not window.actionCamera.isChecked()
    assert window.camera is None and errors


def test_camera_unavailable(window, monkeypatch):
    errors = []
    monkeypatch.setattr(main.QMessageBox, 'critical', lambda *a: errors.append(a))
    monkeypatch.setattr(main, 'open_camera', lambda *a: FakeCamera([], opened=False))
    window.actionCamera.setChecked(True)
    assert not window.actionCamera.isChecked()
    assert window.camera is None and errors


def test_sample_images_do_not_crash(window):
    for name in sorted(os.listdir(IMG_DIR)):
        window.loadImage(os.path.join(IMG_DIR, name))
        before = window.Image.copy()
        window.applemeasurement()
        assert window.label_2.pixmap() is not None
        assert np.array_equal(window.Image, before)
