import numpy as np

import filtering


def naive_konvolusi(image, kernel):
    h, w = kernel.shape[0] // 2, kernel.shape[1] // 2
    out = np.zeros(image.shape)
    for i in range(h, image.shape[0] - h):
        for j in range(w, image.shape[1] - w):
            out[i, j] = np.sum(kernel * image[i - h:i + h + 1, j - w:j + w + 1])
    return out


def test_konvolusi_matches_naive_loop():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (20, 25)).astype(float)
    for kernel in (filtering.GAUSS_5X5, filtering.SOBEL_X, filtering.SOBEL_Y):
        assert np.allclose(filtering.konvolusi(img, kernel), naive_konvolusi(img, kernel))


def test_konvolusi_image_smaller_than_kernel():
    assert not filtering.konvolusi(np.ones((3, 3)), filtering.GAUSS_5X5).any()


def test_konvolusi2_top_left_anchor():
    img = np.arange(16, dtype=float).reshape(4, 4)
    out = filtering.konvolusi2(img, np.ones((2, 2)))
    assert out[0, 0] == 0 + 1 + 4 + 5
    assert not out[3, :].any() and not out[:, 3].any()


def test_threshold_keeps_weak_pixels():
    # bug lama: piksel weak selalu ditimpa 0
    img = np.array([[100, 210, 255]])
    assert filtering.threshold(img, 200, 250, weak_value=200).tolist() == [[0, 200, 255]]


def test_hysteresis_promotes_connected_weak_chain_only():
    img = np.zeros((5, 7), dtype=np.uint8)
    img[2, 1] = 255
    img[2, 2:5] = 200  # rantai weak terhubung ke strong
    img[0, 6] = 200  # weak terisolasi
    out = filtering.hysteresis(img, weak_value=200)
    assert out[2, 1:5].tolist() == [255] * 4
    assert out[0, 6] == 0


def test_canny_detects_square_edges():
    img = np.zeros((60, 60), dtype=np.uint8)
    img[20:40, 20:40] = 255
    stages = filtering.canny(img)
    assert list(stages) == ['Reduksi Noise', 'Gradien', 'Non-Maximum Suppression',
                            'hysteresis part 1', 'hysteresis part 2']
    edges = stages['hysteresis part 2']
    assert edges.dtype == np.uint8
    assert set(np.unique(edges)) <= {0, 255}
    assert edges.any()
    assert not edges[28:32, 28:32].any()  # bagian dalam kotak bukan tepi
    assert not edges[:5, :5].any()  # latar kosong bukan tepi


def test_canny_flat_image():
    stages = filtering.canny(np.full((30, 30), 128, dtype=np.uint8))
    assert not stages['hysteresis part 2'].any()
