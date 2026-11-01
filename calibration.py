"""Kalibrasi piksel -> cm memakai marker ArUco berukuran fisik diketahui.

Marker dicetak (lihat `save_marker_pdf`) lalu diletakkan di samping apel,
SEJAJAR dengan bagian tengah apel (jarak ke kamera sama). Dari 4 sudut marker
dihitung homografi gambar -> bidang marker dalam cm, sehingga kemiringan
kamera terhadap bidang tersebut ikut dikoreksi.

Kalibrasi marker bisa disimpan ke file (`save_calibration`) lalu dipakai lagi
tanpa marker selama posisi & jarak kamera tidak berubah. Bila tidak ada marker
maupun kalibrasi tersimpan, dipakai skala default (`ScaleCalibration`).
"""
import argparse
import json
from dataclasses import dataclass

import cv2
import numpy as np

ARUCO_DICT = cv2.aruco.DICT_4X4_50
MARKER_ID = 0
DEFAULT_MARKER_SIZE_CM = 5.0
# Skala default = konversi versi awal project (0.026458333 cm/px, setara
# 96 DPI). Hanya perkiraan: tidak memperhitungkan jarak kamera.
DEFAULT_PX_PER_CM = 1 / 0.026458333

_dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
_params = cv2.aruco.DetectorParameters()
# sudut sub-piksel: error kecil di sudut marker membesar saat diekstrapolasi ke apel
_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
_detector = cv2.aruco.ArucoDetector(_dictionary, _params)


@dataclass
class Calibration:
    homography: np.ndarray  # 3x3, koordinat gambar (px) -> bidang marker (cm)
    corners: np.ndarray  # 4x2 sudut marker di gambar (TL, TR, BR, BL)
    marker_size_cm: float
    image_size: tuple = None  # (lebar, tinggi) frame saat kalibrasi
    source: str = 'marker'  # 'marker' (terlihat di frame) atau 'tersimpan'

    @property
    def px_per_cm(self):
        sides = np.linalg.norm(self.corners - np.roll(self.corners, -1, axis=0), axis=1)
        return float(sides.mean() / self.marker_size_cm)

    def to_cm(self, points):
        """Proyeksikan titik gambar (Nx2) ke bidang marker dalam cm."""
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.homography).reshape(-1, 2)

    def for_frame(self, width, height):
        """Sesuaikan ke resolusi frame lain dengan rasio aspek sama (kamera
        sama, resolusi beda). Mengembalikan None bila rasio aspek berbeda."""
        if self.image_size is None or tuple(self.image_size) == (width, height):
            return self
        w0, h0 = self.image_size
        if abs(width / height - w0 / h0) > 0.01:
            return None
        sx, sy = width / w0, height / h0
        H = self.homography @ np.diag([1 / sx, 1 / sy, 1.0])
        return Calibration(H, self.corners * [sx, sy], self.marker_size_cm, (width, height), self.source)


@dataclass
class ScaleCalibration:
    """Skala seragam tanpa koreksi perspektif (dipakai sebagai default)."""
    px_per_cm: float
    source: str = 'default'

    def to_cm(self, points):
        return np.asarray(points, dtype=np.float64).reshape(-1, 2) / self.px_per_cm


def save_calibration(path, calibration):
    """Simpan kalibrasi marker ke file JSON."""
    data = {
        'homography': calibration.homography.tolist(),
        'corners': calibration.corners.tolist(),
        'marker_size_cm': calibration.marker_size_cm,
        'image_size': list(calibration.image_size),
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


def load_calibration(path):
    """Muat kalibrasi tersimpan. Mengembalikan None bila file tidak ada;
    ValueError bila isinya rusak."""
    try:
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        return None
    try:
        H = np.array(data['homography'], dtype=np.float64)
        corners = np.array(data['corners'], dtype=np.float64)
        size = float(data['marker_size_cm'])
        image_size = tuple(int(v) for v in data['image_size'])
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError('File kalibrasi tidak valid: {}'.format(e)) from e
    if H.shape != (3, 3) or corners.shape != (4, 2) or size <= 0 or len(image_size) != 2:
        raise ValueError('File kalibrasi tidak valid')
    return Calibration(H, corners, size, image_size, source='tersimpan')


def detect_marker(frame, marker_size_cm=DEFAULT_MARKER_SIZE_CM):
    """Cari marker ArUco (DICT_4X4_50, id 0) dan kembalikan Calibration,
    atau None jika tidak terlihat."""
    if marker_size_cm <= 0:
        raise ValueError('marker_size_cm harus > 0')
    gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = _detector.detectMarkers(gray)
    if ids is None:
        return None

    for c, marker_id in zip(corners, ids.flatten()):
        if marker_id != MARKER_ID:
            continue
        img_pts = c.reshape(4, 2).astype(np.float64)
        s = marker_size_cm
        cm_pts = np.array([[0, 0], [s, 0], [s, s], [0, s]], dtype=np.float64)
        H = cv2.getPerspectiveTransform(img_pts.astype(np.float32), cm_pts.astype(np.float32))
        return Calibration(homography=H, corners=img_pts, marker_size_cm=marker_size_cm,
                           image_size=(gray.shape[1], gray.shape[0]))
    return None


def describe(calibration):
    """Teks singkat sumber & skala kalibrasi untuk status bar / anotasi."""
    labels = {'marker': 'marker', 'tersimpan': 'tersimpan',
              'default': 'skala default (perkiraan, belum dikalibrasi)'}
    return 'Kalibrasi {} {:.1f} px/cm'.format(labels.get(calibration.source, calibration.source),
                                             calibration.px_per_cm)


def draw_calibration(img, calibration, stale=False):
    """Gambar outline marker (bila terlihat) + info kalibrasi (in-place)."""
    if calibration.source == 'marker':
        color = (0, 165, 255) if stale else (255, 0, 255)
        pts = calibration.corners.astype(np.int32)
        cv2.polylines(img, [pts], True, color, 2)
    else:
        color = (0, 140, 255) if calibration.source == 'default' else (255, 0, 255)
    labels = {'marker': 'marker', 'tersimpan': 'tersimpan', 'default': 'default (perkiraan)'}
    text = 'Skala {} {:.1f} px/cm'.format(labels.get(calibration.source, calibration.source),
                                         calibration.px_per_cm)
    # perkecil teks agar muat di lebar gambar
    scale = 1.3
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, scale, 2)
    if tw > img.shape[1] - 20:
        scale *= (img.shape[1] - 20) / tw
    thickness = 2 if scale >= 1 else 1
    org = (10, 10 + int(th * scale / 1.3) + 5)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_PLAIN, scale, (0, 0, 0), thickness + 2)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_PLAIN, scale, color, thickness)


def marker_image(pixels=400):
    return cv2.aruco.generateImageMarker(_dictionary, MARKER_ID, pixels)


def save_marker_pdf(path, marker_size_cm=DEFAULT_MARKER_SIZE_CM):
    """Simpan marker ke PDF A4 dengan ukuran fisik tepat + penggaris 10 cm
    untuk mengecek skala cetak. Butuh QGuiApplication aktif."""
    from PyQt5.QtCore import QMarginsF, QPointF, QRectF, Qt
    from PyQt5.QtGui import QFont, QImage, QPageLayout, QPageSize, QPainter, QPdfWriter, QPen

    writer = QPdfWriter(path)
    writer.setPageSize(QPageSize(QPageSize.A4))
    writer.setPageMargins(QMarginsF(0, 0, 0, 0), QPageLayout.Millimeter)
    writer.setResolution(1200)
    dots_per_cm = writer.resolution() / 2.54

    def cm(v):
        return v * dots_per_cm

    marker = marker_image(1000)
    qimg = QImage(marker.data, marker.shape[1], marker.shape[0], marker.strides[0], QImage.Format_Grayscale8)

    painter = QPainter(writer)
    try:
        painter.setFont(QFont('Arial', 12))
        painter.drawText(QPointF(cm(2), cm(2)), 'Marker kalibrasi Apple Measurement (ArUco 4x4, id 0)')
        painter.drawText(QPointF(cm(2), cm(2.7)),
                         'Cetak skala 100% (jangan "fit to page"). Sisi kotak hitam = {:g} cm.'.format(marker_size_cm))

        painter.drawImage(QRectF(cm(2), cm(4), cm(marker_size_cm), cm(marker_size_cm)), qimg)

        # penggaris 10 cm untuk verifikasi hasil cetak
        y = cm(4 + marker_size_cm + 2)
        painter.setPen(QPen(Qt.black, cm(0.03)))
        painter.drawLine(QPointF(cm(2), y), QPointF(cm(12), y))
        for i in range(11):
            tick = cm(0.5 if i % 5 == 0 else 0.3)
            painter.drawLine(QPointF(cm(2 + i), y), QPointF(cm(2 + i), y - tick))
        painter.drawText(QPointF(cm(2), y + cm(0.8)),
                         'Penggaris 10 cm: ukur dengan penggaris asli. Bila beda, isi ukuran marker sesuai hasil ukur.')
    finally:
        painter.end()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Buat PDF marker kalibrasi.')
    parser.add_argument('output', nargs='?', default='marker_kalibrasi.pdf')
    parser.add_argument('--size', type=float, default=DEFAULT_MARKER_SIZE_CM, help='sisi marker (cm)')
    args = parser.parse_args()

    from PyQt5.QtWidgets import QApplication
    _app = QApplication([])
    save_marker_pdf(args.output, args.size)
    print('Marker {:g} cm disimpan ke {}'.format(args.size, args.output))
