import numpy as np


def konvolusi(image, kernel):
    """Konvolusi 2D (korelasi, kernel tidak di-flip). Piksel tepi yang tidak
    tercakup kernel bernilai 0. Loop hanya di ukuran kernel, bukan per piksel."""
    image = np.asarray(image, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)
    image_h, image_w = image.shape[:2]
    kernel_h, kernel_w = kernel.shape

    h = kernel_h // 2
    w = kernel_w // 2

    image_conv = np.zeros((image_h, image_w))
    if image_h <= 2 * h or image_w <= 2 * w:
        return image_conv

    inner = image_conv[h:image_h - h, w:image_w - w]
    for k in range(-h, h + 1):
        for l in range(-w, w + 1):
            inner += kernel[h + k, w + l] * image[h + k:image_h - h + k, w + l:image_w - w + l]
    return image_conv


def konvolusi2(image, kernel):
    """Konvolusi 2D dengan anchor di pojok kiri-atas kernel."""
    image = np.asarray(image, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)
    image_h, image_w = image.shape[:2]
    kernel_h, kernel_w = kernel.shape

    image_conv = np.zeros((image_h, image_w))
    out_h = image_h - kernel_h + 1
    out_w = image_w - kernel_w + 1
    if out_h <= 0 or out_w <= 0:
        return image_conv

    for k in range(kernel_h):
        for l in range(kernel_w):
            image_conv[:out_h, :out_w] += kernel[k, l] * image[k:k + out_h, l:l + out_w]
    return image_conv


GAUSS_5X5 = (1.0 / 57) * np.array([
    [0, 1, 2, 1, 0],
    [1, 3, 5, 3, 1],
    [2, 5, 9, 5, 2],
    [1, 3, 5, 3, 1],
    [0, 1, 2, 1, 0],
])

SOBEL_X = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1],
])

SOBEL_Y = np.array([
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1],
])


def non_max_suppression(gradien, angle):
    """Pertahankan piksel yang merupakan maksimum lokal searah gradien."""
    H, W = gradien.shape
    p = np.pad(gradien, 1)
    center = p[1:H + 1, 1:W + 1]

    def shift(di, dj):
        return p[1 + di:H + 1 + di, 1 + dj:W + 1 + dj]

    q = np.full((H, W), 255.0)
    r = np.full((H, W), 255.0)

    # angle 0
    m = (angle < 22.5) | (angle >= 157.5)
    q[m], r[m] = shift(0, 1)[m], shift(0, -1)[m]
    # angle 45
    m = (angle >= 22.5) & (angle < 67.5)
    q[m], r[m] = shift(1, -1)[m], shift(-1, 1)[m]
    # angle 90
    m = (angle >= 67.5) & (angle < 112.5)
    q[m], r[m] = shift(1, 0)[m], shift(-1, 0)[m]
    # angle 135
    m = (angle >= 112.5) & (angle < 157.5)
    q[m], r[m] = shift(-1, -1)[m], shift(1, 1)[m]

    result = np.where((center >= q) & (center >= r), center, 0)
    # piksel tepi tidak punya tetangga lengkap
    result[0, :] = result[-1, :] = 0
    result[:, 0] = result[:, -1] = 0
    return result


def threshold(img, weak, strong, weak_value, strong_value=255):
    """Hysteresis tahap 1: klasifikasi piksel jadi strong / weak / 0."""
    out = np.zeros(img.shape, dtype=np.uint8)
    out[img > weak] = weak_value
    out[img > strong] = strong_value
    return out


def hysteresis(img, weak_value, strong_value=255):
    """Hysteresis tahap 2: piksel weak yang terhubung (8-tetangga) ke piksel
    strong dijadikan strong, sisanya dibuang."""
    strong = img == strong_value
    weak = img == weak_value
    H, W = img.shape
    while True:
        p = np.pad(strong, 1)
        neighbour = np.zeros_like(strong)
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di or dj:
                    neighbour |= p[1 + di:H + 1 + di, 1 + dj:W + 1 + dj]
        grow = weak & neighbour & ~strong
        if not grow.any():
            break
        strong |= grow
    return np.where(strong, strong_value, 0).astype(np.uint8)


def canny(gray, weak=200, strong=250):
    """Canny edge detection manual. Mengembalikan setiap tahap (uint8) agar
    bisa ditampilkan satu per satu."""
    konvgauss = konvolusi(gray, GAUSS_5X5)

    konvx = konvolusi(konvgauss, SOBEL_X)
    konvy = konvolusi(konvgauss, SOBEL_Y)

    gradien = np.sqrt((konvx * konvx) + (konvy * konvy))
    # border yang tidak tercakup kernel Gauss + Sobel bernilai 0 dan akan
    # memunculkan tepi palsu, jadi dibuang
    b = GAUSS_5X5.shape[0] // 2 + SOBEL_X.shape[0] // 2
    gradien[:b, :] = gradien[-b:, :] = 0
    gradien[:, :b] = gradien[:, -b:] = 0
    max_grad = np.max(gradien)
    if max_grad > 0:
        gradien = (gradien / max_grad) * 255

    angle = np.arctan2(konvy, konvx) * 180. / np.pi
    angle[angle < 0] += 180

    nms = non_max_suppression(gradien, angle)
    hyst1 = threshold(nms, weak, strong, weak_value=weak)
    hyst2 = hysteresis(hyst1, weak_value=weak)

    return {
        "Reduksi Noise": np.clip(konvgauss, 0, 255).astype(np.uint8),
        "Gradien": gradien.astype(np.uint8),
        "Non-Maximum Suppression": nms.astype(np.uint8),
        "hysteresis part 1": hyst1,
        "hysteresis part 2": hyst2,
    }
