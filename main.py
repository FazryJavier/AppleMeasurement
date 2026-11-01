# IMPORT LIBRARY
import os
import sys
import time
from collections import Counter
import cv2
import numpy as np
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (QAction, QDoubleSpinBox, QFileDialog, QLabel, QMainWindow, QMessageBox,
                             QToolBar)
from PyQt5.uic import loadUi

# IMPORT FILE
import filtering
from AppleDetector import AppleDetector, DiameterSmoother, measure_apples
from calibration import (DEFAULT_MARKER_SIZE_CM, DEFAULT_PX_PER_CM, ScaleCalibration, describe, detect_marker,
                         draw_calibration, load_calibration, save_calibration, save_marker_pdf)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CALIBRATION_FILE = os.path.join(BASE_DIR, 'calibration.json')
CAMERA_INDEX = 0
FRAME_INTERVAL_MS = 33
# berapa lama kalibrasi terakhir tetap dipakai saat marker sesaat tertutup
CALIBRATION_HOLD_S = 1.0


def format_rupiah(value):
    return 'Rp' + '{:,}'.format(value).replace(',', '.')


def summarize(results):
    """Teks (harga, grade) untuk label GUI dari hasil measure_apples."""
    if not results:
        return format_rupiah(0), 'TIDAK ADA APEL'
    prices = [r['price'] for r in results if r['price'] is not None]
    if len(results) == 1:
        return (format_rupiah(prices[0]) if prices else '-'), results[0]['grade']
    counts = Counter(r['grade'] for r in results)
    grades = ', '.join('{}x {}'.format(n, grade) for grade, n in counts.most_common())
    # jumlah apel sudah terlihat dari rincian grade, label harga cukup totalnya
    return (format_rupiah(sum(prices)) if prices else '-'), grades


REJECT_HINTS = {
    'terpotong tepi': 'geser apel ke dalam frame',
    'tidak bulat': 'pisahkan apel / pakai latar polos',
    'kemungkinan daun': 'diabaikan',
}


def detection_status(results, rejected):
    """Ringkasan hasil deteksi + alasan objek yang ditolak untuk status bar."""
    parts = []
    if any(r.get('at_border') for r in results):
        parts.append('Apel menyentuh tepi frame, ukuran mungkin kurang.')
    if rejected:
        reasons = ', '.join('{} {} ({})'.format(n, reason, REJECT_HINTS.get(reason, ''))
                            for reason, n in rejected.most_common())
        parts.append('Ditolak: {}.'.format(reasons))
    if not results:
        parts.insert(0, 'Tidak ada apel terdeteksi.')
    return ' '.join(parts)


def read_image(path):
    # cv2.imread gagal untuk path dengan karakter non-ASCII di Windows
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def open_camera(index=CAMERA_INDEX):
    backend = cv2.CAP_DSHOW if sys.platform == 'win32' else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return cap


class ShowImage(QMainWindow):
    def __init__(self, calibration_path=CALIBRATION_FILE):
        super(ShowImage, self).__init__()
        self.gambar = None
        self.Image = None
        self.output = None
        self.detector = AppleDetector()
        self.smoother = DiameterSmoother()
        self.calibration = None  # marker terakhir yang terlihat
        self.calibration_time = 0.0
        self.calibration_path = calibration_path
        self.saved_calibration = None
        self.camera = None
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.nextFrame)
        self.fps = 0.0
        self.last_frame_time = None

        loadUi(os.path.join(BASE_DIR, 'GUI.ui'), self)
        self.buildToolbar()
        self.buttonReset.clicked.connect(self.reset)
        self.actionOpen.triggered.connect(self.openClicked)
        self.actionSave.triggered.connect(self.save)
        self.actionCanny_Edge.triggered.connect(self.canny)
        self.buttonApple.clicked.connect(self.applemeasurement)
        self.resetLabels()
        self.loadSavedCalibration()

    def buildToolbar(self):
        toolbar = QToolBar('Kamera & Kalibrasi', self)
        self.addToolBar(toolbar)

        self.actionCamera = QAction('Kamera Live', self, checkable=True)
        self.actionCamera.toggled.connect(self.toggleCamera)
        toolbar.addAction(self.actionCamera)
        toolbar.addSeparator()

        toolbar.addWidget(QLabel(' Ukuran marker (cm): '))
        self.markerSize = QDoubleSpinBox(self)
        self.markerSize.setRange(0.5, 50.0)
        self.markerSize.setSingleStep(0.1)
        self.markerSize.setDecimals(2)
        self.markerSize.setValue(DEFAULT_MARKER_SIZE_CM)
        self.markerSize.valueChanged.connect(self.clearCalibration)
        toolbar.addWidget(self.markerSize)

        toolbar.addSeparator()

        self.actionMask = QAction('Tampilkan Mask', self, checkable=True)
        self.actionMask.toggled.connect(self.toggleMask)
        toolbar.addAction(self.actionMask)

        menu = self.menuBar().addMenu('Kalibrasi')
        self.actionSaveCalibration = QAction('Simpan kalibrasi marker', self)
        self.actionSaveCalibration.triggered.connect(self.saveCalibration)
        menu.addAction(self.actionSaveCalibration)
        self.actionClearCalibration = QAction('Hapus kalibrasi tersimpan', self)
        self.actionClearCalibration.triggered.connect(self.deleteSavedCalibration)
        menu.addAction(self.actionClearCalibration)
        menu.addSeparator()
        actionMarker = QAction('Buat marker (PDF)...', self)
        actionMarker.triggered.connect(self.saveMarker)
        menu.addAction(actionMarker)

    def requireImage(self):
        if self.Image is None:
            QMessageBox.warning(self, 'Belum ada gambar', 'Buka gambar dulu lewat menu File > Open.')
            return False
        return True

    def resetLabels(self):
        self.label_16.setText(format_rupiah(0))
        self.label_18.setText('-')

    def clearCalibration(self):
        self.calibration = None
        self.smoother.reset()

    # -----------------------------------------------------------------------------
    # Kalibrasi tersimpan

    def loadSavedCalibration(self):
        try:
            self.saved_calibration = load_calibration(self.calibration_path)
        except (OSError, ValueError) as e:
            self.saved_calibration = None
            self.statusBar().showMessage('Kalibrasi tersimpan diabaikan: {}'.format(e))
            return
        if self.saved_calibration is not None:
            self.statusBar().showMessage('Kalibrasi tersimpan dimuat ({:.1f} px/cm). Siap mengukur tanpa marker.'
                                         .format(self.saved_calibration.px_per_cm))
        else:
            self.statusBar().showMessage('Belum ada kalibrasi tersimpan: memakai skala default (perkiraan). '
                                         'Untuk hasil akurat, kalibrasi dengan marker lalu Kalibrasi > Simpan.')

    def saveCalibration(self):
        if self.calibration is None:
            QMessageBox.warning(self, 'Kalibrasi', 'Belum ada marker yang terdeteksi. Nyalakan kamera atau buka '
                                'gambar yang memuat marker, lalu klik Apple Measurement terlebih dahulu.')
            return
        try:
            save_calibration(self.calibration_path, self.calibration)
        except OSError as e:
            QMessageBox.critical(self, 'Kalibrasi', 'Gagal menyimpan kalibrasi:\n{}'.format(e))
            return
        self.loadSavedCalibration()
        QMessageBox.information(self, 'Kalibrasi disimpan',
                                'Kalibrasi {:.1f} px/cm disimpan. Selama posisi dan jarak kamera tidak berubah, '
                                'marker boleh disingkirkan.'.format(self.calibration.px_per_cm))

    def deleteSavedCalibration(self):
        try:
            os.remove(self.calibration_path)
        except FileNotFoundError:
            pass
        except OSError as e:
            QMessageBox.critical(self, 'Kalibrasi', 'Gagal menghapus kalibrasi:\n{}'.format(e))
            return
        self.loadSavedCalibration()

    # -----------------------------------------------------------------------------
    # File

    def openClicked(self):
        flname, _ = QFileDialog.getOpenFileName(self, 'Open File', os.path.join(BASE_DIR, 'img'),
                                                "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if flname:
            self.loadImage(flname)

    def save(self):
        image = self.output if self.output is not None else self.Image
        if image is None:
            self.requireImage()
            return
        flname, _ = QFileDialog.getSaveFileName(self, 'Save File', '', "Image Files (*.jpg *.png)")
        if flname:
            ok, buf = cv2.imencode(os.path.splitext(flname)[1] or '.jpg', image)
            if ok:
                buf.tofile(flname)
            else:
                QMessageBox.critical(self, 'Gagal', 'Gambar gagal disimpan.')

    def saveMarker(self):
        flname, _ = QFileDialog.getSaveFileName(self, 'Simpan Marker', 'marker_kalibrasi.pdf', "PDF (*.pdf)")
        if flname:
            save_marker_pdf(flname, self.markerSize.value())
            QMessageBox.information(self, 'Marker disimpan',
                                    'Cetak {} dengan skala 100%, lalu ukur sisi kotak hitam dengan penggaris '
                                    'dan isi nilainya di "Ukuran marker".'.format(os.path.basename(flname)))

    def loadImage(self, flname):
        image = read_image(flname)
        if image is None:
            QMessageBox.critical(self, 'Gagal', 'File bukan gambar yang valid:\n{}'.format(flname))
            return
        self.actionCamera.setChecked(False)
        self.Image = image
        self.gambar = flname
        self.output = None
        self.displayimage(1)
        self.label_2.clear()
        self.resetLabels()

    def reset(self):
        self.output = None
        self.smoother.reset()
        self.label_2.clear()
        self.resetLabels()
        if self.gambar is not None and self.camera is None:
            self.Image = read_image(self.gambar)

    # -----------------------------------------------------------------------------
    # F2 Canny Edge Detection
    def canny(self):
        if not self.requireImage():
            return
        gray = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        for title, stage in filtering.canny(gray).items():
            cv2.imshow(title, stage)

    # -----------------------------------------------------------------------------
    # Pengukuran

    def currentCalibration(self, frame, live):
        """Kalibrasi untuk frame ini -> (kalibrasi, stale). Prioritas: marker
        di frame -> kalibrasi tersimpan -> skala default."""
        now = time.monotonic()
        found = detect_marker(frame, self.markerSize.value())
        if found is not None:
            self.calibration = found
            self.calibration_time = now
            return found, False
        if self.calibration is not None and live and now - self.calibration_time <= CALIBRATION_HOLD_S:
            return self.calibration, True

        if self.saved_calibration is not None:
            saved = self.saved_calibration.for_frame(frame.shape[1], frame.shape[0])
            if saved is not None:
                return saved, False
        return ScaleCalibration(DEFAULT_PX_PER_CM), False

    def process(self, frame, live=False):
        calibration, stale = self.currentCalibration(frame, live)
        annotated, results = measure_apples(frame, calibration, self.detector,
                                            self.smoother if live else None)
        draw_calibration(annotated, calibration, stale)

        price_text, grade_text = summarize(results)
        self.label_16.setText(price_text)  # Label utk mencantumkan harga pada GUI
        self.label_18.setText(grade_text)  # Label utk mencantumkan grade pada GUI

        status = describe(calibration) + '.'
        status += ' ' + detection_status(results, self.detector.last_rejected)
        if live:
            status += '  {:.0f} fps'.format(self.fps)
        self.statusBar().showMessage(status)

        if self.actionMask.isChecked():
            cv2.imshow('Mask Apel', self.detector.last_mask)

        self.output = annotated
        self.displayimage(2, annotated)
        return results

    def applemeasurement(self):
        if self.camera is not None:
            # saat live, tombol ini membekukan frame terakhir beserta hasilnya
            self.actionCamera.setChecked(False)
            return
        if not self.requireImage():
            return
        for i, r in enumerate(self.process(self.Image), 1):
            print('#{} {} {} cm -> {} ({})'.format(i, r['color'], r['diameter'], r['grade'], r['price']))
        print(self.statusBar().currentMessage())

    # -----------------------------------------------------------------------------
    # Kamera live

    def toggleCamera(self, on):
        if on:
            self.startCamera()
        else:
            self.stopCamera()

    def startCamera(self):
        cap = open_camera()
        if not cap.isOpened():
            cap.release()
            self.actionCamera.setChecked(False)
            QMessageBox.critical(self, 'Kamera', 'Kamera tidak bisa dibuka. Pastikan tidak dipakai aplikasi lain.')
            return
        self.camera = cap
        self.gambar = None
        self.smoother.reset()
        self.last_frame_time = None
        self.buttonApple.setText('Ambil Foto')
        self.timer.start(FRAME_INTERVAL_MS)

    def stopCamera(self):
        self.timer.stop()
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        self.buttonApple.setText('Apple Measurement')

    def nextFrame(self):
        if self.camera is None:
            return
        ok, frame = self.camera.read()
        if not ok or frame is None:
            self.actionCamera.setChecked(False)
            QMessageBox.critical(self, 'Kamera', 'Gagal membaca frame dari kamera.')
            return

        now = time.monotonic()
        if self.last_frame_time is not None:
            dt = now - self.last_frame_time
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt) if dt > 0 else self.fps
        self.last_frame_time = now

        self.Image = frame
        self.displayimage(1)
        self.process(frame, live=True)

    def toggleMask(self, on):
        if not on:
            try:
                cv2.destroyWindow('Mask Apel')
            except cv2.error:
                pass

    def closeEvent(self, event):
        self.stopCamera()
        cv2.destroyAllWindows()
        super(ShowImage, self).closeEvent(event)

    def displayimage(self, windows, image=None):
        image = self.Image if image is None else image
        image = np.ascontiguousarray(image)
        qformat = QImage.Format_Indexed8

        if len(image.shape) == 3:
            if (image.shape[2]) == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888

        # rgbSwapped() membuat salinan, jadi buffer numpy aman setelah ini
        img = QImage(image.data, image.shape[1], image.shape[0], image.strides[0], qformat).rgbSwapped()

        label = self.label if windows == 1 else self.label_2
        # pertahankan rasio aspek agar bentuk apel tidak penyok di layar
        pixmap = QPixmap.fromImage(img).scaled(label.size(), QtCore.Qt.KeepAspectRatio,
                                               QtCore.Qt.SmoothTransformation)
        label.setScaledContents(False)
        label.setPixmap(pixmap)
        label.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = ShowImage()
    window.setWindowTitle('Pengolahan Citra')
    window.show()
    sys.exit(app.exec_())
