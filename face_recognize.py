import cv2, sys, numpy, os

haar_file = 'haarcascade_frontalface_default.xml'
datasets = 'datasets'

# Create a list of images and a list of corresponding names
(images, labels, names, id) = ([], [], {}, 0)
for (subdirs, dirs, files) in os.walk(datasets):
    for subdir in dirs:
        names[id] = subdir
        subjectpath = os.path.join(datasets, subdir)
        for filename in os.listdir(subjectpath):
            path = os.path.join(subjectpath, filename)  # Use os.path.join for path concatenation
            label = id
            images.append(cv2.imread(path, cv2.IMREAD_GRAYSCALE))  # Load as grayscale
            labels.append(int(label))
        id += 1

(width, height) = (130, 100)

# Convert images and labels lists into NumPy arrays
labels = numpy.array(labels, dtype=numpy.int32)

# Create the model
model = cv2.face.LBPHFaceRecognizer_create()
model.train(images, labels)

confidence_threshold = 60  # Set your desired confidence threshold

face_cascade = cv2.CascadeClassifier(haar_file)
webcam = cv2.VideoCapture(0)
while True:
    ret, im = webcam.read()  # Capture frame
    if not ret:  # If the webcam stream ends
        break
    
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (width, height))
        
        # Try to recognize the face
        prediction = model.predict(face_resize)
        confidence = prediction[1]
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 3)

        if confidence < confidence_threshold:
            recognized_name = names[prediction[0]]
            cv2.putText(im, f'{recognized_name} - Confidence: {confidence:.2f}', (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
        else:
            cv2.putText(im, 'Unrecognized', (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))  # Red text for unknown faces
            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 3)  # Red rectangle for unknown faces

    cv2.imshow('OpenCV', im)

    # Check if the OpenCV window is closed
    if cv2.getWindowProperty('OpenCV', cv2.WND_PROP_VISIBLE) < 1:
        break

    key = cv2.waitKey(10)
    if key == 27:
        break

webcam.release()
cv2.destroyAllWindows()

