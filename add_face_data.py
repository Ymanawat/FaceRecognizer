import cv2
import sys
import os

# Path to the Haar cascade file for face detection
haar_file = 'haarcascade_frontalface_default.xml'

# Folder where face data will be stored
datasets = 'datasets'

# Initialize OpenCV's face detector and webcam
face_cascade = cv2.CascadeClassifier(haar_file)
webcam = cv2.VideoCapture(0)

# Function to create a directory for a new person
def create_person_directory(name):
    #Creates a directory for storing face data of a new person.
    path = os.path.join(datasets, name)
    if not os.path.isdir(path):
        os.mkdir(path)
    return path

# Check if the person's name is provided as a command-line argument
if len(sys.argv) < 2:
    print("Please provide the person's name as a command-line argument.")
    sys.exit(1)
name = sys.argv[1]

# Create a directory for the person
person_path = create_person_directory(name)

# Capture and store faces
count = 1
while count <= 120:
    # Read a frame from the webcam
    (_, im) = webcam.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 4)
    
    for (x, y, w, h) in faces:
        # Draw a rectangle around the detected face
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face = gray[y:y + h, x:x + w]
        
        # Save the face image in the person's directory
        cv2.imwrite(os.path.join(person_path, f'{count}.png'), face)
        count += 1
        
    # Display the image with detected faces
    cv2.imshow('OpenCV', im)
    
    # Check for the 'Esc' key press to exit
    key = cv2.waitKey(10)
    if key == 27:
        break

# Release the webcam and close all OpenCV windows
webcam.release()
cv2.destroyAllWindows()
