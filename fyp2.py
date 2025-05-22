#!/usr/bin/env python
# coding: utf-8

# In[5]:


# ✅ 1. Import Libraries
import tensorflow as tf
import os
import numpy as np

# ✅ 2. Set Dataset Path and Parameters
data_dir = "C:\\Users\\PCD\\Desktop\\rtrain"  # Ensure this folder contains one subfolder per person
img_height = 64
img_width = 64
batch_size = 32
epochs = 20
seed = 123
val_split = 0.2

# ✅ 3. Load Dataset
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=val_split,
    subset="training",
    seed=seed,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=val_split,
    subset="validation",
    seed=seed,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# ✅ 4. Save Class Names
class_names = train_ds.class_names
print("Detected classes:", class_names)

with open("class_names.txt", "w") as f:
    for name in class_names:
        f.write(name + "\n")

# ✅ 5. Optimize Dataset Performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# ✅ 6. Data Augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])

# ✅ 7. Build the Model
model = tf.keras.Sequential([
    data_augmentation,
    tf.keras.layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),

    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(128, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(class_names), activation='softmax')
])

# ✅ 8. Compile and Train
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.build(input_shape=(None, img_height, img_width, 3))
model.summary()

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

# ✅ 9. Save Model as .h5
model.save("face_recognition_modell.h5")
print("✅ Saved model as face_recognition_modell.h5")

# ✅ 10. Convert to TFLite and save as .tflite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open("face_recognition_modell.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Saved model as face_recognition_modell.tflite")


# In[ ]:


#!/usr/bin/env python
# coding: utf-8

import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

# Load class names
with open("class_names.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# ✅ Load TFLite model with correct name
interpreter = tflite.Interpreter(model_path="face_recognition_modell.tflite")
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

confidence_threshold = 0.4

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (320, 240))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_roi = frame[y:y+h, x:x+w]
        face_gray = gray[y:y+h, x:x+w]

        face_img = cv2.resize(face_roi, (64, 64))
        face_img = face_img.astype(np.float32) / 255.0
        input_data = np.expand_dims(face_img, axis=0)

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])

        class_id = np.argmax(prediction)
        confidence = np.max(prediction)

        print(f"Predicted: {class_names[class_id]}, Confidence: {confidence:.2f}")

        if confidence >= confidence_threshold:
            label = class_names[class_id]
            color = (0, 255, 0)
        else:
            label = "Unknown"
            color = (0, 0, 255)

        smiles = smile_cascade.detectMultiScale(face_gray, scaleFactor=1.8, minNeighbors=20)
        smile_label = "Smiling" if len(smiles) > 0 else "Not Smiling"

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"{label}: {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(frame, smile_label, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.imshow("Face Recognition (TFLite)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# In[ ]:


import tensorflow as tf

# Load the trained Keras model (with double "l")
model = tf.keras.models.load_model("face_recognition_modell.h5")

# Create a converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# No quantization (keeps float32 precision)
tflite_model = converter.convert()

# Save the .tflite file with "2" at the end
with open("face_recognition_modell2.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Model converted and saved as face_recognition_modell2.tflite")

