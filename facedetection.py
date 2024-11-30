import cv2

# Load the Cascade Classifier for Face Detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Capture Video from Device (e.g., Webcam)
cap = cv2.VideoCapture(0)

while True:
    # Read Frame from Video
    ret, img = cap.read()
    
    # Convert to Grayscale for Detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect Faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    # Draw Rectangle around Faces
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        
    # Display Output
    cv2.imshow('Face Detection - OpenCV', img)
    
    # Exit on Key Press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release Resources
cap.release()
cv2.destroyAllWindows()
