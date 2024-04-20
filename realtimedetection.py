import cv2
import numpy as np
from keras.models import load_model
import sys
sys.stdout.reconfigure(encoding='utf-8')


# Load the model
model = load_model("facialemotionmodel.h5")

# Load the Haar cascade for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Define emotion labels
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

# Function to extract features from the image
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Start the webcam
webcam = cv2.VideoCapture(0)

# Set the window title
cv2.setWindowTitle("Real-Time Emotion Detection", "Real-Time Emotion Detection")

while True:
    # Capture frame-by-frame
    ret, frame = webcam.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    try:
        for (p, q, r, s) in faces:
            # Extract the face region from the frame
            image = gray[q:q + s, p:p + r]
            
            # Resize the face region to match the model input size
            image = cv2.resize(image, (48, 48))
            
            # Extract features from the resized face region
            img = extract_features(image)
            
            # Make prediction using the model
            pred = model.predict(img)
            prediction_label = labels[pred.argmax()]
            
            # Draw rectangle around the face
            cv2.rectangle(frame, (p, q), (p + r, q + s), (255, 0, 0), 2)
            
            # Display the predicted emotion label
            cv2.putText(frame, '%s' % (prediction_label), (p - 10, q - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2,
                        (0, 0, 255))
        
        # Display the frame
        cv2.imshow("Real-Time Emotion Detection", frame)
        
        # Check for exit key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    except cv2.error:
        pass

# Release the webcam and close all OpenCV windows
webcam.release()
cv2.destroyAllWindows()
