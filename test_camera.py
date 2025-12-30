from picamera2 import Picamera2
import cv2
import time
import numpy as np

# Inisialisasi kamera
picam2 = Picamera2()
# Kita coba set secara eksplisit ke RGB888
config = picam2.create_still_configuration(main={"format": "RGB888", "size": (640, 480)})
picam2.configure(config)
picam2.start()

print("⏳ Menunggu AWB (White Balance) stabil selama 2 detik...")
time.sleep(2)

# Ambil satu frame
image = picam2.capture_array()

# --- EKSPERIMEN 1: Simpan apa adanya ---
# OpenCV menganggap array yang masuk adalah BGR. 
# Jika aslinya RGB, maka di file ini warnanya akan BIRU (Avatar).
cv2.imwrite("foto_as_is.jpg", image)

# --- EKSPERIMEN 2: Paksa Konversi RGB ke BGR ---
# Jika aslinya RGB, setelah dikonversi ke BGR, OpenCV akan menyimpannya dengan warna NORMAL.
image_converted = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
cv2.imwrite("foto_converted.jpg", image_converted)

picam2.stop()

print("✅ Selesai! Silakan cek dua file di folder Anda:")
print("1. foto_as_is.jpg")
print("2. foto_converted.jpg")
print("\nManakah yang warnanya normal?")
