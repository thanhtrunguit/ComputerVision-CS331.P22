
import cv2
import numpy as np
from ultralytics import YOLO
import tensorflow as tf
import time
import torch
import models.inception_resnet_v1 as inception_resnet_v1
import matplotlib.pyplot as plt
# ─── CONFIG ────────────────────────────────────────────────────────────────
YOLO_MODEL_PATH      = "/Users/thanhtrung/UIT/AdComVision/face_recognition/faceRecognition/models/yolov8m-face.pt"
FACENET_WEIGHT_PATH  = "/Users/thanhtrung/UIT/AdComVision/face_recognition/faceRecognition/models/facenet_keras_weights.h5"
TRUNG_VECTOR_PATH    = "/Users/thanhtrung/UIT/AdComVision/face_recognition/faceRecognition/database/trung_vector.npy"
THIEN_VECTOR_PATH    = "/Users/thanhtrung/UIT/AdComVision/face_recognition/faceRecognition/database/thien_face_vector.npy"
SIMILARITY_THRESHOLD = 0.8
DETECT_CONFIDENCE    = 0.8
IMG_SIZE             = 640

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(device)

gpus = tf.config.list_physical_devices("GPU")
if not gpus:
    raise RuntimeError("No GPU found—did you install tensorflow-metal?")
print("Available GPU devices:", gpus)
# ─── LOAD FACE RECOGNITION MODEL ─────────────────────────────────────────
# 1) Instantiate & load FaceNet (Inception-ResNet-V1)
# facenet = inception_resnet_v1.InceptionResNetV1()
# facenet.load_weights(FACENET_WEIGHT_PATH)

with tf.device("/GPU:0"):
    facenet = inception_resnet_v1.InceptionResNetV1()
    facenet.load_weights(FACENET_WEIGHT_PATH)
    # You can print a summary if you like:
    # facenet.summary()

# 2) Utility: preprocess for FaceNet
def preprocess_for_facenet(bgr_roi):
    rgb = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (160, 160)).astype("float32")
    normed  = (resized - resized.mean()) / resized.std()
    return np.expand_dims(normed, axis=0)  # shape (1,160,160,3)

# ─── LOAD STORED GALLERY VECTORS ──────────────────────────────────────────
trung_vec = np.load(TRUNG_VECTOR_PATH)   # shape (128,)
thien_vec = np.load(THIEN_VECTOR_PATH)   # shape (128,)

trung_vec = trung_vec[0]
thien_vec = thien_vec[0]


# Normalize once for faster dot-product sim
trung_vec /= np.linalg.norm(trung_vec)
thien_vec /= np.linalg.norm(thien_vec)


# ─── INITIALIZE YOLOv8 FACE DETECTOR ─────────────────────────────────────

# YOLOv8 detector
detector = YOLO(YOLO_MODEL_PATH).to(device)
detector.model.fuse().half()

# ─── START WEBCAM LOOP ───────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
prev_fps = 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    start_time = time.time()
    # frame = cv2.flip(frame, 1)
    # 1) Run YOLOv8 detection
    results = detector(frame, imgsz=IMG_SIZE, conf=DETECT_CONFIDENCE)

    # 2) For each detection, crop, embed, compare
    for r in results:
        for box, conf in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
            x1, y1, x2, y2 = box.astype(int)
            crop = frame[y1:y2, x1:x2]


            inp  = preprocess_for_facenet(crop)

            # 3) Get embedding
            with tf.device("/GPU:0"):
                emb = facenet.predict(inp)[0]

            # emb = facenet.predict(inp)[0]
            emb /= np.linalg.norm(emb)

            # 4) Cosine similarities
            sim_trung = float(np.dot(trung_vec, emb))
            sim_thien = float(np.dot(thien_vec, emb))

            # 5) Determine best match
            if sim_trung >= SIMILARITY_THRESHOLD or sim_thien >= SIMILARITY_THRESHOLD:
                if sim_trung > sim_thien:
                    label, sim = "Trung", sim_trung
                else:
                    label, sim = "Thien", sim_thien
            else:
                label, sim = "Unknown", max(sim_trung, sim_thien)

            # 6) Draw
            text = f"{label} {sim:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, text, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    end_time = time.time()
    fps = 1.0 / (end_time - start_time)
    prev_fps = 0.9 * prev_fps + 0.1 * fps
    cv2.putText(frame, f"FPS: {prev_fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    # 7) Show
    cv2.imshow("YOLOv8 + FaceNet Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
