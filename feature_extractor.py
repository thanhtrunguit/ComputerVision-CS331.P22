import cv2


from keras_facenet import FaceNet
import ssl
import numpy as np

# ssl._create_default_https_context = ssl._create_unverified_context

#Step 1
import models.inception_resnet_v1 as inception_resnet_v1

#Step 2
model = inception_resnet_v1.InceptionResNetV1()
model.load_weights('/Users/thanhtrung/UIT/AdComVision/face_recognition/faceRecognition/models/facenet_keras_weights.h5')


img_path = "/Users/thanhtrung/UIT/AdComVision/face_recognition/faceRecognition/database/thien_face_crop.jpg"
output_path = '/Users/thanhtrung/UIT/AdComVision/face_recognition/faceRecognition/database/thien_face_vector.npy'


img = cv2.imread(img_path)
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
face = cv2.resize(rgb, (160, 160)).astype("float32")
face = (face - face.mean()) / face.std()
faces = np.expand_dims(face, axis=0)

# Get embeddings
embs = model.predict(faces)
trung_vector = embs[0]

np.save(output_path, embs)  # Saves trung_vector → trung_vector.npy :contentReference[oaicite:2]{index=2}

print(f"Vector saved to: {output_path}")

print("Embedding shape:", embs.shape)
print("First embedding vector:", embs[0])