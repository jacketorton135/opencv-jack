import tensorflow as tf
import cv2
import numpy as np
import os

# Define image path and get directories (categories)
IMAGEPATH = 'images'  # Modify to your image directory
dirs = os.listdir(IMAGEPATH)

# Set image dimensions and category count
h, w = 224, 224
category = len(dirs)

# Load the entire model (architecture + weights)
try:
    model = tf.keras.models.load_model("G:/opencv/model_complete.h5")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Start webcam capture
cap = cv2.VideoCapture(1)

# Check if the webcam is accessible
if not cap.isOpened():
    print("Error: Could not access the camera.")
    exit()

# Start the webcam feed loop
try:
    while True:
        ret, img = cap.read()
        
        # Check if the frame is read correctly
        if not ret:
            print("Error: Failed to capture image.")
            break

        # Resize and preprocess the image for model input
        resized = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
        image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        image = image / 255.0  # Normalize pixel values to [0, 1]
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

        # Model prediction
        predict = model.predict(image)
        i = np.argmax(predict[0])
        str1 = dirs[i] + "   " + str(predict[0][i])

        # Display the predicted class on the frame
        img = cv2.putText(img, str1, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Camera Feed', img)  # Show the camera feed in a window

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(50) & 0xFF == ord('q'):
            break
finally:
    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()