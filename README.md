# Apple Measurement

Desktop app (PyQt5 + OpenCV) that determines the **grade and selling price of
apples from their diameter**, using image files or a **live laptop webcam**.

## How it works

```
image / camera frame
  → 1. detect apples    (HSV colour segmentation + shape filters)
  → 2. measure diameter (pixels → cm using calibration)
  → 3. look up grade & price in the table
```

1. **Detection**: HSV colour segmentation (red, yellow and green apples) → fill
   holes left by specular highlights → remove stems (morphological opening) →
   keep round shapes only (circularity, solidity). Leaves and objects cut off
   by the frame edge are ignored.
2. **Diameter**: the longest side of the apple contour's `minAreaRect`,
   converted to cm with the calibration (see below). In live mode the value is
   stabilised with a median over the last 15 frames per apple.
3. **Grade & price** (`grade_apple` in `AppleDetector.py`). When several apples
   are detected, prices are summed:

| Diameter | Grade | Price |
|---|---|---|
| 7 – 8 cm | PREMIUM | Rp12.200 |
| 6 – <7 cm | REGULER (regular) | Rp9.500 |
| 5 – <6 cm | SUPER SAVER | Rp7.500 |
| < 5 cm | TERLALU KECIL (too small) | Rp0 |
| > 8 cm | TIDAK VALID (out of range) | - |

## Calibration (pixels → cm)

An apple's size in pixels depends on the camera distance, so a scale is
needed. The app picks one automatically in this order of priority (the source
in use is shown in the status bar and the top-left corner of the output image):

| Priority | Source | Used when | Accuracy |
|---|---|---|---|
| 1 | **marker** | An ArUco marker is visible in the frame | Accurate (corrects distance and tilt) |
| 2 | **saved** | `calibration.json` exists and the frame has the same resolution/aspect ratio | Accurate as long as the camera position and distance are unchanged |
| 3 | **default scale** | Neither of the above | Estimate: 37.8 px/cm (the original project's 96 DPI conversion) |

The app therefore **works out of the box without calibration** (default
scale). For physically correct results, calibrate **once** and save it.

## Installation (Windows)

Requires Python 3.9+.

```bash
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python main.py
```

## Usage

The app's interface is in Indonesian; menu and button names are given below as
they appear in the app, with a translation in brackets.

### A. From an image file (no calibration)

**File > Open** → choose an image from `img/` → click **Apple Measurement**.
Price and grade appear next to the output panel using the default scale. The
sample images carry no distance information, so the values are estimates.

### B. Live camera — calibrate once, then no marker needed

1. Print [`marker_kalibrasi.pdf`](marker_kalibrasi.pdf) at **100% scale**
   (turn off "fit to page"). Measure the black square with a ruler; if it is
   not exactly 5 cm, enter the measured value in **Ukuran marker (cm)**
   (marker size) on the toolbar.
2. Mount the camera in a **fixed** position (e.g. the laptop on a desk, with
   apples always placed on a marked spot on paper/cardboard at the same
   distance from the camera).
3. Click **Kamera Live** (live camera). Place the marker where the apple will
   go, **at the height of the apple's centre** (camera from above → raise it
   ±3–4 cm; camera from the front → stand the marker upright). A purple
   outline means the marker was detected.
4. Click **Kalibrasi > Simpan kalibrasi marker** (Calibration > Save marker
   calibration). It is stored in `calibration.json` and loaded automatically
   every time the app starts.
5. Remove the marker. Put an apple on the same spot on a plain background →
   price and grade appear immediately. **Ambil Foto** (take photo) freezes the
   result and **File > Save** saves the annotated image.

If the camera is moved or the distance changes, repeat steps 3–4 (or use
**Kalibrasi > Hapus kalibrasi tersimpan** (delete saved calibration) to go back
to the default scale). A marker visible in the frame always takes priority, so
it can also stay next to the apple for the most accurate measurement.

**Tampilkan Mask** (show mask) on the toolbar opens the segmentation mask
window, to check whether the background or lighting interferes with detection.

To generate a marker of another size, use **Kalibrasi > Buat marker (PDF)**
(create marker) or:

```bash
.venv\Scripts\python calibration.py marker.pdf --size 8
```

## Tests

```bash
.venv\Scripts\python -m pip install -r requirements-dev.txt
.venv\Scripts\python -m pytest
```

## Limitations

- The default scale is only an estimate; the real diameter depends on the
  camera distance.
- A saved calibration is only valid for the camera position and apple distance
  used during calibration, and for frames with the same aspect ratio.
- Colour-based detection needs a plain background that is not strongly red,
  yellow or green. Touching apples of the same colour are read as one non-round
  object and skipped. When an object is rejected, the reason is shown in the
  status bar (cut off by the edge / not round / probably a leaf).
- Leaves attached by a stem or lying next to an apple are ignored, but a leaf
  covering the apple's body is included in the measurement. A green object much
  smaller (< 60%) than another apple and within 10% of its diameter is treated
  as a leaf.
- An apple slightly touching the frame edge is still measured and marked
  "(tepi)" (edge); its size may be underestimated.
- Marker accuracy: on synthetic tests the error is ≤ 0.1 cm (frontal) and
  ≤ 0.15 cm (tilted camera). Not yet benchmarked with a real camera against
  apples measured with a ruler. Lens distortion is not corrected; keep apples
  near the centre of the frame.
