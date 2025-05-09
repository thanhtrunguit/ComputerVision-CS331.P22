import numpy as np
from numpy.linalg import norm

# Load the feature vectors
vec1 = np.load("/Users/thanhtrung/UIT/AdComVision/face_recognition/faceRecognition/database/thien_face_vector.npy")
vec2 = np.load("/Users/thanhtrung/UIT/AdComVision/face_recognition/faceRecognition/database/trung_vector.npy")

# Compute cosine similarity
cos_sim = np.dot(vec1[0], vec2[0]) / (norm(vec1[0]) * norm(vec2[0]))

print("Cosine Similarity between thien and trung:", cos_sim)


import cv2
import time
import torch
import numpy as np
from ultralytics import YOLO
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
import ssl
import matplotlib.pyplot as plt
ssl._create_default_https_context = ssl._create_unverified_context

# —————————————————————————————
# CONFIGURATION
# —————————————————————————————
YOLO_MODEL_PATH = "/Users/thanhtrung/UIT/AdComVision/face_recognition/faceRecognition/models/yolov8m-face.pt"
DETECTION_IMG_SIZE = 640
CONFIDENCE_THRESH = 0.8
FEATURE_VECTOR_PATH = "/Users/thanhtrung/UIT/AdComVision/face_recognition/faceRecognition/trung_01.npy"
SIMILARITY_THRESHOLD = 0.7
RESNET_WEIGHTS_PATH = "/Users/thanhtrung/UIT/AdComVision/face_recognition/faceRecognition/models/resnet50-0676ba61.pth"

# —————————————————————————————
# INITIALIZE DETECTORS & FEATURE EXTRACTOR
# —————————————————————————————
device = "mps" if torch.backends.mps.is_available() else "cpu"

# YOLOv8 detector
detector = YOLO(YOLO_MODEL_PATH).to(device)
detector.model.model.half()

# ResNet50 feature extractor (remove final classification layer)
resnet = models.resnet50()
resnet.fc = torch.nn.Identity()
# Load custom weights
state = torch.load(RESNET_WEIGHTS_PATH, map_location=device)
resnet.load_state_dict(state, strict=False)
resnet = resnet.to(device).eval()

# Preprocessing for ResNet50
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
])

# Load stored feature vector
stored_vec = np.load(FEATURE_VECTOR_PATH)
# stored_vec = stored_vec / np.linalg.norm(stored_vec)

# —————————————————————————————
# REAL-TIME LOOP WITH FPS & RECOGNITION
# —————————————————————————————
cap = cv2.VideoCapture(0)
prev_fps = 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    start_time = time.time()

    # Run detection
    results = detector(
        frame,
        stream=True,
        imgsz=DETECTION_IMG_SIZE,
        conf=CONFIDENCE_THRESH,
        device=device
    )

    # Parse detections
    for r in results:
        for box, conf in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
            x1, y1, x2, y2 = box.astype(int)
            w, h = x2 - x1, y2 - y1
            if w < 20 or h < 20:
                continue

            # Crop face region
            crop = frame[y1:y2, x1:x2]

            # Extract features via ResNet50
            try:
                inp = preprocess(crop).unsqueeze(0).to(device)
                with torch.no_grad():
                    feat = resnet(inp).cpu().numpy()[0]
                # feat = feat / np.linalg.norm(feat)

                # Compute cosine similarity
                # sim = cosine_similarity(feat.reshape(1, -1), stored_vec.reshape(1, -1))[0][0]

                sim = np.dot(stored_vec, feat) / (np.linalg.norm(stored_vec) * np.linalg.norm(feat))
                label = "Trung" if sim >= SIMILARITY_THRESHOLD else "Unknown"
            except Exception as e:
                sim = 0.0
                label = "Error"

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f"{label} {sim:.2f}"
            cv2.putText(frame, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Compute and display FPS
    end_time = time.time()
    fps = 1.0 / (end_time - start_time)
    prev_fps = 0.9 * prev_fps + 0.1 * fps
    cv2.putText(frame, f"FPS: {prev_fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

    # Show output
    cv2.imshow("YOLOv8 + ResNet50 Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()

