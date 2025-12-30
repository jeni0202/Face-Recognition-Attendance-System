import csv
import cv2
import os
import numpy as np

# -------------------------------
def is_number(s):
    try:
        float(s)
        return True
    except:
        return False


# -------------------------------
# Cloud-safe take image function
def takeImages(image, Id, name):
    if not (is_number(Id) and name.isalpha()):
        return "ID must be numeric and Name must be alphabetic"

    # Convert image to OpenCV format
    frame = np.array(image)

    if frame is None or frame.size == 0:
        return "Invalid image"

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    harcascadePath = "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(harcascadePath)

    faces = detector.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return "No face detected. Please try again."

    os.makedirs("TrainingImage", exist_ok=True)
    os.makedirs("StudentDetails", exist_ok=True)

    sampleNum = 0
    for (x, y, w, h) in faces:
        sampleNum += 1
        face_img = gray[y:y + h, x:x + w]

        img_path = f"TrainingImage/{name}.{Id}.{sampleNum}.jpg"
        cv2.imwrite(img_path, face_img)

    # Save student details
    csv_path = "StudentDetails/StudentDetails.csv"
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, "a", newline="") as csvFile:
        writer = csv.writer(csvFile)
        if not file_exists:
            writer.writerow(["Id", "Name"])
        writer.writerow([Id, name])

    return f"Images saved for ID: {Id}, Name: {name}"
