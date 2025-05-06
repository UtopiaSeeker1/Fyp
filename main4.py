#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

# Load class names
with open("class_names.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# Load TFLite model
interpreter = tflite.Interpreter(model_path="face_recognition_modell.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Haar face and smile detectors
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# Video capture
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

if not cap.isOpened():
    print("❌ Webcam not detected.")
    exit()

# Thresholds and smoothing
confidence_threshold = 0.4
smile_counter = 0
smile_persistence = 5

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (320, 240))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(60, 60))

    for (x, y, w, h) in faces:
        # Padding around face box
        padding = 10
        x1 = max(x - padding, 0)
        y1 = max(y - padding, 0)
        x2 = min(x + w + padding, frame.shape[1])
        y2 = min(y + h + padding, frame.shape[0])

        face_roi = frame[y1:y2, x1:x2]
        face_gray = gray[y1:y2, x1:x2]

        # Preprocess face for model
        face_img = cv2.resize(face_roi, (64, 64))
        face_img = face_img.astype(np.float32) / 255.0
        input_data = np.expand_dims(face_img, axis=0)

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])

        class_id = np.argmax(prediction)
        id_conf = np.max(prediction)

        if id_conf >= confidence_threshold:
            label = class_names[class_id]
            color = (0, 255, 0)
        else:
            label = "Unknown"
            color = (0, 0, 255)

        print(f"Predicted: {label}, Confidence: {id_conf:.2f}")

        # Smile detection with smoothing
        smiles = smile_cascade.detectMultiScale(face_gray, scaleFactor=1.8, minNeighbors=20)

        if len(smiles) > 0:
            smile_counter += 1
        else:
            smile_counter = max(smile_counter - 1, 0)

        smile_label = "Smiling" if smile_counter >= smile_persistence else "Not Smiling"

        # Draw everything
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label}: {id_conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(frame, smile_label, (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.imshow("Face Recognition + Smile Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

