import datetime
import os
import time
import cv2
import pandas as pd
import numpy as np


def recognize_attendence(image):
    """
    image: PIL Image or numpy array captured using st.camera_input()
    """

    # ---------- Load trained model ----------
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
    except AttributeError:
        recognizer = cv2.face.createLBPHFaceRecognizer()

    model_path = "TrainingImageLabel/Trainner.yml"
    if not os.path.isfile(model_path):
        raise FileNotFoundError("Trained model not found. Please train images first.")

    recognizer.read(model_path)

    # ---------- Load face detector ----------
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    # ---------- Load student details ----------
    csv_path = "StudentDetails/StudentDetails.csv"
    if not os.path.isfile(csv_path):
        raise FileNotFoundError("StudentDetails.csv not found.")

    df = pd.read_csv(csv_path)

    # ---------- Convert image ----------
    frame = np.array(image)
    if frame is None or frame.size == 0:
        raise ValueError("Invalid image")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=5, minSize=(50, 50)
    )

    col_names = ["Id", "Name", "Date", "Time"]
    attendance = pd.DataFrame(columns=col_names)

    for (x, y, w, h) in faces:
        face_img = gray[y:y + h, x:x + w]

        Id, conf = recognizer.predict(face_img)
        confidence = round(100 - conf)

        if confidence > 67:
            name = df.loc[df["Id"] == Id]["Name"].values
            name = name[0] if len(name) > 0 else "Unknown"

            ts = time.time()
            date = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
            timeStamp = datetime.datetime.fromtimestamp(ts).strftime("%H:%M:%S")

            attendance.loc[len(attendance)] = [Id, name, date, timeStamp]

            label = f"{Id}-{name} ({confidence}%)"
            color = (0, 255, 0)
        else:
            label = "Unknown"
            color = (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            frame,
            label,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
        )

    attendance = attendance.drop_duplicates(subset=["Id"], keep="first")

    # ---------- Save attendance ----------
    if not attendance.empty:
        os.makedirs("Attendance", exist_ok=True)
        ts = time.time()
        date = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
        timeStamp = datetime.datetime.fromtimestamp(ts).strftime("%H-%M-%S")
        fileName = f"Attendance/Attendance_{date}_{timeStamp}.csv"
        attendance.to_csv(fileName, index=False)

    return frame, attendance
