from collections import Counter, deque
from dataclasses import dataclass

import cv2
import numpy as np

# (diameter minimum cm, diameter maksimum cm, harga, grade), urut dari besar
GRADES = [
    (7.0, 8.0, 12200, 'PREMIUM'),
    (6.0, 7.0, 9500, 'REGULER'),
    (5.0, 6.0, 7500, 'SUPER SAVER'),
]
NOT_CALIBRATED = 'BELUM KALIBRASI'

# Rentang warna kulit apel di HSV OpenCV (H 0-179). Saturasi minimum dibuat
# tinggi agar kulit tangan, kayu, dan latar pucat tidak ikut.
# (nama warna, (H, S, V) bawah, (H, S, V) atas)
COLOR_BANDS = [
    ('MERAH', (0, 100, 40), (12, 255, 255)),
    ('MERAH', (160, 100, 40), (179, 255, 255)),
    ('KUNING', (12, 150, 90), (20, 255, 255)),  # oranye/semburat Gala & Fuji
    ('KUNING', (20, 110, 90), (35, 255, 255)),
    ('HIJAU', (35, 70, 40), (85, 255, 255)),
]


def grade_apple(diameter_cm):
    """Tentukan (harga, grade) dari diameter apel dalam cm.

    Rentang dibuat kontinu: 7-8 PREMIUM, 6-<7 REGULER, 5-<6 SUPER SAVER,
    <5 TERLALU KECIL, >8 di luar kategori (harga None)."""
    if diameter_cm < GRADES[-1][0]:
        return 0, 'TERLALU KECIL'
    for low, high, price, grade in GRADES:
        if low <= diameter_cm <= high:
            return price, grade
    return None, 'TIDAK VALID'


def color_mask(hsv, bands=COLOR_BANDS):
    """Mask biner gabungan seluruh rentang warna apel + mask per warna."""
    per_color = {}
    for name, low, high in bands:
        m = cv2.inRange(hsv, low, high)
        per_color[name] = cv2.bitwise_or(per_color[name], m) if name in per_color else m
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for m in per_color.values():
        mask = cv2.bitwise_or(mask, m)
    return mask, per_color


def _ellipse(size):
    size = max(3, int(size) | 1)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


@dataclass
class DetectedApple:
    contour: np.ndarray  # kontur bersih (tanpa tangkai, lubang terisi)
    color: str
    at_border: bool = False  # menyentuh tepi frame: ukuran mungkin kurang


class AppleDetector:
    """Segmentasi apel berbasis warna HSV + filter bentuk.

    Tahapan: blur -> mask HSV (merah/kuning/hijau) -> closing -> isi lubang
    (pantulan cahaya) -> opening per objek (buang tangkai) -> filter luas,
    kebulatan (circularity), solidity. Blob yang tidak bulat (mis. apel
    menempel daun/pisang) dicoba lagi per warna. Objek kecil yang menempel
    pada apel jauh lebih besar dianggap daun.

    Asumsi: apel diletakkan terpisah (tidak menempel) di atas latar polos
    yang warnanya tidak merah/kuning/hijau mencolok."""

    def __init__(self, min_diameter_ratio=0.05, min_circularity=0.6, min_solidity=0.88,
                 max_border_ratio=0.1, leaf_size_ratio=0.6):
        self.min_diameter_ratio = min_diameter_ratio
        self.min_circularity = min_circularity
        self.min_solidity = min_solidity
        self.max_border_ratio = max_border_ratio
        self.leaf_size_ratio = leaf_size_ratio
        self.last_mask = None
        self.last_rejected = Counter()

    def detect(self, frame):
        h, w = frame.shape[:2]
        base = min(h, w)
        min_diameter = max(20.0, base * self.min_diameter_ratio)
        close_kernel = _ellipse(base / 80)

        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask, per_color = color_mask(hsv)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

        rejected = Counter()
        candidates = []
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            if 2 * cv2.minEnclosingCircle(c)[1] < min_diameter:
                continue
            contour, reason = self._accept(c, min_diameter, w, h)
            if contour is not None:
                candidates.append(contour)
                continue
            # coba pisahkan per warna: apel merah yang menempel daun hijau /
            # pisang kuning jadi blob tersendiri di mask warnanya
            parts = self._split_by_color(c, per_color, close_kernel, min_diameter, w, h)
            if parts:
                candidates.extend(parts)
            else:
                rejected[reason] += 1

        apples = []
        clean = np.zeros_like(mask)
        for contour in candidates:
            if self._is_leaf(contour, candidates):
                rejected['kemungkinan daun'] += 1
                continue
            cv2.drawContours(clean, [contour], -1, 255, -1)
            apples.append(DetectedApple(contour, self._dominant_color(contour, per_color),
                                        _border_ratio(contour, w, h) > 0))

        self.last_mask = clean
        self.last_rejected = rejected
        return apples

    def detect_objects(self, frame):
        return [a.contour for a in self.detect(frame)]

    def _accept(self, contour, min_diameter, width, height):
        """Bersihkan kontur lalu uji -> (kontur, None) bila lolos, atau
        (None, alasan penolakan)."""
        contour = self._clean_object(contour)
        if contour is None:
            return None, 'tidak bulat'
        if _border_ratio(contour, width, height) > self.max_border_ratio:
            return None, 'terpotong tepi'
        if not self._is_round(contour, min_diameter):
            return None, 'tidak bulat'
        return contour, None

    def _split_by_color(self, contour, per_color, close_kernel, min_diameter, width, height):
        region = np.zeros(next(iter(per_color.values())).shape, dtype=np.uint8)
        cv2.drawContours(region, [contour], -1, 255, -1)
        parts = []
        for m in per_color.values():
            sub = cv2.morphologyEx(cv2.bitwise_and(m, region), cv2.MORPH_CLOSE, close_kernel)
            cnts, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                if 2 * cv2.minEnclosingCircle(c)[1] < min_diameter:
                    continue
                accepted, _ = self._accept(c, min_diameter, width, height)
                if accepted is not None:
                    parts.append(accepted)
        return parts

    def _is_leaf(self, contour, candidates):
        """Objek yang jauh lebih kecil dan menempel pada apel lain hampir
        pasti daun/tangkai apel tersebut."""
        (x, y), r = cv2.minEnclosingCircle(contour)
        for other in candidates:
            if other is contour:
                continue
            (ox, oy), orad = cv2.minEnclosingCircle(other)
            if r >= self.leaf_size_ratio * orad:
                continue
            gap = np.hypot(x - ox, y - oy) - r - orad
            if gap <= 0.1 * 2 * orad:  # menempel / celah < 10% diameter
                return True
        return False

    @staticmethod
    def _clean_object(contour):
        """Isi lubang lalu opening dengan kernel ~15% diameter objek agar
        tangkai & tonjolan tipis hilang. Mengembalikan kontur terbesar."""
        x, y, w, h = cv2.boundingRect(contour)
        pad = 2
        roi = np.zeros((h + 2 * pad, w + 2 * pad), dtype=np.uint8)
        cv2.drawContours(roi, [contour - [x - pad, y - pad]], -1, 255, -1)
        equiv_d = 2 * np.sqrt(cv2.contourArea(contour) / np.pi)
        roi = cv2.morphologyEx(roi, cv2.MORPH_OPEN, _ellipse(equiv_d * 0.15))
        cnts, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts:
            return None
        return max(cnts, key=cv2.contourArea) + [x - pad, y - pad]

    def _is_round(self, contour, min_diameter):
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        hull_area = cv2.contourArea(cv2.convexHull(contour))
        if perimeter == 0 or hull_area == 0:
            return False
        if 2 * np.sqrt(area / np.pi) < min_diameter:
            return False
        circularity = 4 * np.pi * area / perimeter ** 2
        solidity = area / hull_area
        return circularity >= self.min_circularity and solidity >= self.min_solidity

    @staticmethod
    def _dominant_color(contour, per_color):
        region = np.zeros(next(iter(per_color.values())).shape, dtype=np.uint8)
        cv2.drawContours(region, [contour], -1, 255, -1)
        counts = {name: cv2.countNonZero(cv2.bitwise_and(m, region)) for name, m in per_color.items()}
        return max(counts, key=counts.get)


def _border_ratio(contour, width, height, margin=2):
    """Porsi titik kontur yang menempel tepi frame (0 = tidak menyentuh).
    Porsi besar berarti apel terpotong sehingga diameternya tidak valid."""
    pts = contour.reshape(-1, 2)
    on_border = ((pts[:, 0] <= margin) | (pts[:, 1] <= margin) |
                 (pts[:, 0] >= width - 1 - margin) | (pts[:, 1] >= height - 1 - margin))
    return float(on_border.mean())


# Nama lama dipertahankan agar kode yang sudah ada tetap jalan
HomogeneousBgDetector = AppleDetector


def measure_diameter_cm(contour, calibration):
    """Diameter = sisi terpanjang minAreaRect kontur setelah diproyeksikan ke
    bidang marker (cm). Proyeksi mengoreksi skala & kemiringan kamera."""
    pts_cm = calibration.to_cm(contour.reshape(-1, 2)).astype(np.float32)
    (_, _), (w, h), _ = cv2.minAreaRect(pts_cm)
    return round(float(max(w, h)), 1)


def measure_apples(img, calibration=None, detector=None, smoother=None):
    """Deteksi & ukur semua apel pada gambar BGR.

    Mengembalikan (gambar beranotasi, list hasil). Setiap hasil berisi
    'diameter' (cm, None bila belum kalibrasi), 'price', 'grade', 'color' dan
    'center' (px). `smoother` (DiameterSmoother) dipakai pada mode kamera
    live. Gambar input tidak diubah."""
    detector = detector or AppleDetector()
    annotated = img.copy()
    results = []

    apples = detector.detect(img)
    for apple in apples:
        (x, y), (w, h), _ = cv2.minAreaRect(apple.contour)
        if calibration is not None:
            diameter = measure_diameter_cm(apple.contour, calibration)
            price, grade = grade_apple(diameter)
        else:
            diameter, price, grade = None, None, NOT_CALIBRATED
        results.append({'diameter': diameter, 'price': price, 'grade': grade,
                        'color': apple.color, 'center': (float(x), float(y)),
                        'at_border': apple.at_border,
                        'size_px': float(max(w, h))})

    if smoother is not None:
        results = smoother.update(results, (img.shape[1], img.shape[0]))
    annotate(annotated, [a.contour for a in apples], results)
    return annotated, results


def annotate(img, contours, results):
    """Gambar kontur apel dan teks diameter/grade pada img (in-place)."""
    cv2.drawContours(img, contours, -1, (255, 0, 0), 2)
    for i, r in enumerate(results, 1):
        x, y = r['center']
        size = r.get('size_px', 150)
        scale = min(2.0, max(0.8, size / 150))
        thickness = 2 if scale >= 1.2 else 1
        size_text = '{} cm'.format(r['diameter']) if r['diameter'] is not None else '? cm'
        if r.get('at_border'):
            size_text += ' (tepi)'
        cv2.circle(img, (int(x), int(y)), 4, (0, 0, 255), -1)
        for text, dy in (('#{} {}'.format(i, size_text), -10), (r['grade'], 15)):
            (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, scale, thickness)
            org = (int(x - tw / 2), int(y + dy * scale))
            cv2.putText(img, text, org, cv2.FONT_HERSHEY_PLAIN, scale, (0, 0, 0), thickness + 2)
            cv2.putText(img, text, org, cv2.FONT_HERSHEY_PLAIN, scale, (100, 255, 0), thickness)


class DiameterSmoother:
    """Menstabilkan pengukuran kamera live: setiap apel dilacak dari posisi
    pusatnya antar-frame dan diameternya diambil median N frame terakhir."""

    def __init__(self, window=15, max_jump_ratio=0.15, max_missed=10):
        self.window = window
        self.max_jump_ratio = max_jump_ratio
        self.max_missed = max_missed
        self.tracks = []  # dict: center, history, missed

    def update(self, results, frame_size):
        max_jump = self.max_jump_ratio * max(frame_size)
        unmatched = list(self.tracks)
        smoothed = []
        for r in results:
            track = min(unmatched, key=lambda t: _dist(t['center'], r['center']), default=None)
            if track is None or _dist(track['center'], r['center']) > max_jump:
                track = {'history': deque(maxlen=self.window)}
                self.tracks.append(track)
            else:
                unmatched.remove(track)
            track['center'] = r['center']
            track['missed'] = 0
            if r['diameter'] is not None:
                track['history'].append(r['diameter'])

            out = dict(r)
            if track['history']:
                out['diameter'] = round(float(np.median(track['history'])), 1)
                out['price'], out['grade'] = grade_apple(out['diameter'])
            smoothed.append(out)

        for t in unmatched:
            t['missed'] += 1
        self.tracks = [t for t in self.tracks if t['missed'] <= self.max_missed]
        return smoothed

    def reset(self):
        self.tracks = []


def _dist(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))
