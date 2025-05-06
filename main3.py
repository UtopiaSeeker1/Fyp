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

# Load CNN-based DNN face detector
# Make sure deploy.prototxt and res10_300x300_ssd_iter_140000.caffemodel are in the same folder
face_net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")

# Load smile detector (Haar cascade)
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

    frame_resized = cv2.resize(frame, (320, 240))
    h, w = frame_resized.shape[:2]

    # Face detection with DNN
    blob = cv2.dnn.blobFromImage(cv2.resize(frame_resized, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x1, y1) = box.astype("int")

            face_roi = frame_resized[y:y1, x:x1]
            face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

            # Preprocess face for TFLite model
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
            cv2.rectangle(frame_resized, (x, y), (x1, y1), color, 2)
            cv2.putText(frame_resized, f"{label}: {id_conf:.2f}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame_resized, smile_label, (x, y1 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.imshow("Face Recognition + Smile Detection (DNN)", frame_resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

