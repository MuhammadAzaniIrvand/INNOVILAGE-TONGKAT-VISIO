import os
import sys
import argparse
import time
import cv2
import numpy as np
from ultralytics import YOLO

# ==========================================
# 1. KONFIGURASI VISIO
# ==========================================
CUSTOM_THRESHOLDS = {
    'got': 0.15, 'tiang': 0.20, 'tangga menaik': 0.25, 'tangga_menurun': 0.25,
    'orang': 0.45, 'kendaraan': 0.45, 'meja': 0.60, 'kursi': 0.60,
    'pintu_terbuka': 0.55, 'pintu_tertutup': 0.55
}
DEFAULT_THRESH = 0.40
MAX_BOX_HEIGHT_RATIO = 0.85 

np.random.seed(42)
BBOX_COLORS = np.random.randint(0, 255, size=(100, 3), dtype=np.uint8).tolist()

# ==========================================
# 2. SETUP ARGUMEN
# ==========================================
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='best_v6_ncnn_model', help='Path to model')
parser.add_argument('--imgsz', type=int, default=320, help='Inference size')
parser.add_argument('--duration', type=int, default=20, help='Durasi rekam (detik)')
args = parser.parse_args()

model_path = args.model
imgsz = args.imgsz
record_duration = args.duration

# ==========================================
# 3. INISIALISASI MODEL & WARMUP
# ==========================================
print(f"🚀 HEADLESS MODE | Model: {model_path}")

if not os.path.exists(model_path):
    print(f"❌ Error: Model {model_path} tidak ditemukan.")
    sys.exit(1)

# Load model
model = YOLO(model_path, task='detect')
labels = model.names

# --- CARA 1: WARMUP RUN (Mencegah Segfault di awal) ---
print("⏳ Warming up NCNN model...")
dummy_frame = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
model.predict(dummy_frame, imgsz=imgsz, verbose=False)
print("✅ Model Ready.")

# ==========================================
# 4. INISIALISASI KAMERA
# ==========================================
try:
    from picamera2 import Picamera2
    cap = Picamera2()
    config = cap.create_preview_configuration(
        main={"size": (640, 480), "format": "BGR888"},
        controls={"AfMode": 2, "AfRange": 0}
    )
    cap.configure(config)
    cap.start()
    time.sleep(2) # Beri waktu sensor stabil
    print("📷 Kamera OK (Native Pi 5)")
except Exception as e:
    print(f"❌ Gagal init kamera: {e}")
    sys.exit(1)

# INIT RECORDER
output_file = 'hasil_headless.avi'
recorder = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'MJPG'), 20, (640,480))
print(f"🔴 Merekam ke: {output_file} selama {record_duration} detik...")

# ==========================================
# 5. LOOP UTAMA
# ==========================================
avg_fps = 0
fps_buffer = []
frame_count = 0
SKIP_FRAME = 2 
start_time = time.time()

try:
    while True:
        t_start = time.perf_counter()
        
        if (time.time() - start_time) > record_duration:
            print("\n⏰ Waktu habis.")
            break

        frame = cap.capture_array()
        frame_count += 1
        
        if frame_count % SKIP_FRAME == 0 or frame_count == 1:
            # Inference dengan mode CPU eksplisit untuk stabilitas
            results = model.predict(frame, imgsz=imgsz, conf=0.1, verbose=False)
            detections = results[0].boxes
            
            for box in detections:
                # --- CARA 2: HANDLING NaN & KOORDINAT AMAN ---
                coords = box.xyxy[0].tolist() # Ambil koordinat
                
                # Cek jika ada nilai NaN (Not a Number)
                if any(np.isnan(coords)):
                    continue
                
                try:
                    # Konversi ke int dengan aman
                    x1, y1, x2, y2 = map(int, coords)
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                except (ValueError, IndexError):
                    continue

                class_name = labels[cls_id]
                thresh = CUSTOM_THRESHOLDS.get(class_name, DEFAULT_THRESH)
                
                if conf < thresh: continue
                
                # Filter Box Tinggi
                box_h = y2 - y1
                if (box_h / frame.shape[0]) > MAX_BOX_HEIGHT_RATIO:
                    if class_name not in ['pintu_terbuka', 'pintu_tertutup', 'orang']:
                        continue 

                # Visualisasi
                color = BBOX_COLORS[cls_id % len(BBOX_COLORS)]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                label = f"{class_name} {int(conf*100)}%"
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                text_y = y1 - 5 if y1 > 20 else y1 + h + 5
                text_x = max(0, min(x1, frame.shape[1] - w))
                
                cv2.rectangle(frame, (text_x, text_y - h - 4), (text_x + w, text_y + 4), (0,0,0), -1)
                cv2.putText(frame, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

        # Hitung FPS
        t_stop = time.perf_counter()
        fps = 1 / (t_stop - t_start)
        fps_buffer.append(fps)
        if len(fps_buffer) > 30: fps_buffer.pop(0)
        avg_fps = np.mean(fps_buffer)

        cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        recorder.write(frame)

        if frame_count % 30 == 0:
            print(f"Running... FPS: {avg_fps:.1f} | Frame: {frame_count}")

except KeyboardInterrupt:
    print("\nInterrupted by user.")
except Exception as e:
    print(f"\n❌ Error mendadak: {e}")

finally:
    print("Cleanup...")
    if 'cap' in locals(): cap.stop()
    if 'recorder' in locals(): recorder.release()
    print(f"✅ Selesai! File tersimpan: {output_file}")
