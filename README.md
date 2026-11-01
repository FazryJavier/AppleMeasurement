# Apple Measurement

Aplikasi desktop (PyQt5 + OpenCV) untuk menentukan **grade dan harga jual buah
apel berdasarkan diameternya**, dari file gambar atau **kamera laptop secara live**.

## Cara kerja

```
gambar / frame kamera
  → 1. deteksi apel   (segmentasi warna HSV + filter bentuk)
  → 2. ukur diameter  (piksel → cm memakai kalibrasi)
  → 3. tentukan grade & harga dari tabel
```

1. **Deteksi**: segmentasi warna HSV (apel merah, kuning, hijau) → isi lubang
   pantulan cahaya → buang tangkai (opening) → filter bentuk bulat
   (circularity, solidity). Daun dan objek terpotong tepi frame diabaikan.
2. **Diameter**: sisi terpanjang `minAreaRect` dari kontur apel, dikonversi ke
   cm dengan kalibrasi (lihat di bawah). Pada mode live nilai distabilkan
   dengan median 15 frame terakhir per apel.
3. **Grade & harga** (`grade_apple` di `AppleDetector.py`). Jika ada beberapa
   apel, harga ditotal:

| Diameter | Grade | Harga |
|---|---|---|
| 7 – 8 cm | PREMIUM | Rp12.200 |
| 6 – <7 cm | REGULER | Rp9.500 |
| 5 – <6 cm | SUPER SAVER | Rp7.500 |
| < 5 cm | TERLALU KECIL | Rp0 |
| > 8 cm | TIDAK VALID | - |

## Kalibrasi (piksel → cm)

Ukuran apel dalam piksel tergantung jarak kamera, jadi perlu skala. Aplikasi
memilih otomatis dengan urutan prioritas berikut (sumbernya tampil di status
bar dan pojok kiri atas gambar hasil):

| Prioritas | Sumber | Kapan dipakai | Akurasi |
|---|---|---|---|
| 1 | **marker** | Marker ArUco terlihat di frame | Akurat (koreksi jarak & kemiringan) |
| 2 | **tersimpan** | Ada `calibration.json` dan resolusi/rasio frame sama | Akurat selama posisi & jarak kamera tidak berubah |
| 3 | **skala default** | Tidak ada keduanya | Perkiraan: 37,8 px/cm (= konversi versi awal project, 96 DPI) |

Jadi aplikasi **langsung bisa dipakai tanpa kalibrasi** (skala default). Untuk
hasil yang benar secara fisik, cukup kalibrasi **sekali** lalu simpan.

## Instalasi (Windows)

Butuh Python 3.9+.

```bash
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python main.py
```

## Pemakaian

### A. Dari file gambar (tanpa kalibrasi)

**File > Open** → pilih gambar dari `img/` → klik **Apple Measurement**. Harga &
grade tampil di panel kanan memakai skala default. Gambar contoh tidak punya
informasi jarak, jadi angkanya perkiraan.

### B. Kamera live — kalibrasi sekali, lalu tanpa marker

1. Cetak [`marker_kalibrasi.pdf`](marker_kalibrasi.pdf) dengan **skala 100%**
   (matikan "fit to page"). Ukur sisi kotak hitam dengan penggaris; bila bukan
   5 cm, isi hasil ukurnya di **Ukuran marker (cm)** pada toolbar.
2. Pasang kamera di posisi **tetap** (mis. laptop di meja, apel selalu diletakkan
   di titik bertanda pada kertas/kardus berjarak sama dari kamera).
3. Klik **Kamera Live**. Letakkan marker di tempat apel nanti diletakkan,
   **setinggi bagian tengah apel** (kamera dari atas → ganjal ±3–4 cm; kamera
   dari depan → marker berdiri tegak). Outline ungu = marker terdeteksi.
4. Klik **Kalibrasi > Simpan kalibrasi marker**. Tersimpan di `calibration.json`
   dan otomatis dimuat setiap aplikasi dibuka.
5. Singkirkan marker. Letakkan apel di titik yang sama di atas latar polos →
   harga & grade langsung tampil. **Ambil Foto** membekukan hasil, **File > Save**
   menyimpan gambar beranotasi.

Jika kamera dipindah atau jaraknya berubah, ulangi langkah 3–4 (atau
**Kalibrasi > Hapus kalibrasi tersimpan** untuk kembali ke skala default).
Marker yang terlihat di frame selalu didahulukan, jadi marker juga boleh
dibiarkan di samping apel untuk pengukuran paling akurat.

**Tampilkan Mask** di toolbar membuka jendela mask segmentasi, untuk mengecek
apakah latar/pencahayaan mengganggu deteksi.

## Test

```bash
.venv\Scripts\python -m pip install -r requirements-dev.txt
.venv\Scripts\python -m pytest
```

## Keterbatasan

- Skala default hanya perkiraan; diameter sebenarnya bergantung jarak kamera.
- Kalibrasi tersimpan hanya berlaku untuk posisi kamera & jarak apel saat
  kalibrasi, dan untuk frame dengan rasio aspek yang sama.
- Deteksi berbasis warna: butuh latar polos yang tidak merah/kuning/hijau
  mencolok. Apel yang saling menempel (warna sama) terbaca satu objek tidak
  bulat sehingga dilewati. Jika objek ditolak, alasannya tampil di status bar
  (terpotong tepi / tidak bulat / kemungkinan daun).
- Daun yang tersambung tangkai atau menempel di samping apel diabaikan, tapi
  daun yang menimpa badan apel ikut terukur. Objek hijau yang jauh lebih kecil
  (< 60%) dan berjarak < 10% diameter dari apel lain dianggap daun.
- Apel yang sedikit menyentuh tepi frame tetap diukur dengan tanda "(tepi)";
  ukurannya bisa lebih kecil dari aslinya.
- Akurasi marker: pada uji sintetis error ≤ 0,1 cm (tampak lurus) dan
  ≤ 0,15 cm (kamera miring). Dengan kamera asli belum diuji memakai apel +
  penggaris. Distorsi lensa tidak dikoreksi; letakkan apel dekat tengah frame.
