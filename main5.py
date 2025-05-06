#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

# Load class names
with open("class_names.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# Load improved (non-quantized) TFLite model
interpreter = tflite.Interpreter(model_path="face_recognition_modell2.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load DNN face detector
face_net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")

# Load Haar cascade for smile detection
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# Open camera
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
if not cap.isOpened():
    print("❌ Webcam not detected.")
    exit()

# Parameters
confidence_threshold = 0.5
smile_counter = 0
smile_persistence = 3
frame_width, frame_height = 320, 240

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_resized = cv2.resize(frame, (frame_width, frame_height))
    h, w = frame_resized.shape[:2]

    # Face detection
    blob = cv2.dnn.blobFromImage(cv2.resize(frame_resized, (300, 300)), 1.0, (300, 300), (104, 177, 123))
    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.6:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x, y, x1, y1 = box.astype("int")

            x, y = max(0, x), max(0, y)
            x1, y1 = min(w, x1), min(h, y1)

            face_roi = frame_resized[y:y1, x:x1]
            face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

            # Prepare face for recognition
            face_input = cv2.resize(face_roi, (64, 64))
            face_input = face_input.astype(np.float32) / 255.0
            face_input = np.expand_dims(face_input, axis=0)

            interpreter.set_tensor(input_details[0]['index'], face_input)
            interpreter.invoke()
            prediction = interpreter.get_tensor(output_details[0]['index'])

            class_id = int(np.argmax(prediction))
            id_conf = float(np.max(prediction))

            if id_conf >= confidence_threshold:
                label = class_names[class_id]
                color = (0, 255, 0)
            else:
                label = "Unknown"
                color = (0, 0, 255)

            print(f"Predicted: {label}, Confidence: {id_conf:.2f}")

            # Smile detection
            smiles = smile_cascade.detectMultiScale(face_gray, scaleFactor=1.8, minNeighbors=20, minSize=(25, 25))
            if len(smiles) > 0:
                smile_counter += 1
            else:
                smile_counter = max(smile_counter - 1, 0)

            smile_label = "Smiling" if smile_counter >= smile_persistence else "Not Smiling"

            # Draw bounding box and labels
            cv2.rectangle(frame_resized, (x, y), (x1, y1), color, 2)
            cv2.putText(frame_resized, f"{label}: {id_conf:.2f}", (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            cv2.putText(frame_resized, smile_label, (x, y1 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

    cv2.imshow("Face Recognition + Smile Detection (Enhanced)", frame_resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

