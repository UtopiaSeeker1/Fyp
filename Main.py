#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

# Load class names
with open("class_names.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# Load TFLite model
interpreter = tflite.Interpreter(model_path="face_recognition_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']

# Load Haar cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# Start video capture
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

if not cap.isOpened():
    print("❌ Webcam not detected.")
    exit()

confidence_threshold = 0.8

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame for performance (optional)
    frame = cv2.resize(frame, (320, 240))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_roi = frame[y:y+h, x:x+w]
        face_gray = gray[y:y+h, x:x+w]

        # Preprocess for model input
        face_img = cv2.resize(face_roi, (64, 64))  # same as model input size
        face_img = face_img.astype(np.float32) / 255.0
        input_data = np.expand_dims(face_img, axis=0)

        # Set tensor and run inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])

        class_id = np.argmax(prediction)
        confidence = np.max(prediction)

        if confidence >= confidence_threshold:
            label = class_names[class_id]
            color = (0, 255, 0)
        else:
            label = "Unknown"
            color = (0, 0, 255)

        # Smile detection
        smiles = smile_cascade.detectMultiScale(face_gray, scaleFactor=1.8, minNeighbors=20)
        smile_label = "Smiling" if len(smiles) > 0 else "Not Smiling"

        # Draw results
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"{label}: {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(frame, smile_label, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.imshow("Face Recognition (TFLite)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# In[ ]:




