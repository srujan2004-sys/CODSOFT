import cv2
import numpy as np

# Load the pre-trained Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Known faces (just using image paths for now)
known_faces = {
    "Student 1": "student1.jpg",  # replace with the actual path to your image
    "Student 2": "student2.jpg"   # replace with the actual path to your image
}

# Simple face recognizer (using LBPH method from OpenCV)
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Function to detect faces in an image
def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert the image to grayscale
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))  # Detect faces
    return faces, gray

# Train the face recognizer using known faces
def train_recognizer():
    labels = []
    faces = []
    
    for label, image_path in known_faces.items():
        img = cv2.imread(image_path)  # Read the image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        detected_faces, _ = detect_faces(img)  # Detect faces in the image

        for (x, y, w, h) in detected_faces:  # Loop through each detected face
            face = gray[y:y+h, x:x+w]  # Extract the face region
            faces.append(face)
            labels.append(label)  # Save the label for this face
    
    recognizer.train(faces, np.array(labels))  # Train the model with the collected faces and labels

# Train recognizer before starting detection
train_recognizer()

# Function to recognize faces in an image
def recognize_face(image):
    faces, gray = detect_faces(image)  # Detect faces in the image
    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]  # Get the face region
        label, confidence = recognizer.predict(face)  # Predict the label for this face

        print(f"Detected: {label} with confidence: {confidence}")

        # Draw a rectangle around the face and add a label
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)  # Draw rectangle
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)  # Put label

    cv2.imshow('Face Recognition', image)  # Display the image with faces and labels
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Use the webcam to capture real-time video feed
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()  # Read frame from webcam
    if not ret:
        break

    recognize_face(frame)  # Recognize faces in the frame

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()
