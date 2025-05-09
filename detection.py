import cv2
from ultralytics import YOLO

# Constants
YOLO_MODEL_PATH = "/Users/thanhtrung/UIT/AdComVision/face_recognition/faceRecognition/models/yolov6m-face.pt"
IMAGE_PATH = "/Users/thanhtrung/UIT/AdComVision/face_recognition/faceRecognition/thien_03.jpg"
CROPPED_FACE_PATH = "/Users/thanhtrung/UIT/AdComVision/face_recognition/faceRecognition/database/thien_face_crop_test.jpg"

# Load image
image = cv2.imread(IMAGE_PATH)
if image is None:
    raise FileNotFoundError(f"Could not read image from {IMAGE_PATH}")

# Load YOLOv8 face detection model
model = YOLO(YOLO_MODEL_PATH)

# Run detection
results = model(image, imgsz=640, conf=0.8)

# Parse first detected face (you can modify to handle multiple)
for r in results:
    boxes = r.boxes.xyxy.cpu().numpy()
    if len(boxes) == 0:
        raise ValueError("No faces detected.")

    # Assume the most confident detection is the first
    x1, y1, x2, y2 = map(int, boxes[0])
    face_crop = image[y1:y2, x1:x2]

    # Save cropped image
    cv2.imwrite(CROPPED_FACE_PATH, face_crop)
    print(f"Cropped face saved at: {CROPPED_FACE_PATH}")
    break
