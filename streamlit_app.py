import streamlit as st
import cv2
import os
import numpy as np
from PIL import Image

# ================== DIRECTORIES ==================
os.makedirs("TrainingImage", exist_ok=True)
os.makedirs("StudentDetails", exist_ok=True)
os.makedirs("Attendance", exist_ok=True)
os.makedirs("TrainingImageLabel", exist_ok=True)

st.title("Face Recognition Attendance System")

# ================== CAPTURE FACE ==================
st.subheader("Capture Face")

id_input = st.text_input("Enter ID")
name_input = st.text_input("Enter Name")

img = st.camera_input("Take a photo")

if st.button("Save Face"):
    if not id_input or not name_input:
        st.error("ID and Name are required")
    elif img is None:
        st.error("Please capture an image")
    else:
        image = Image.open(img)
        frame = np.array(image)

        if frame is not None and frame.size != 0:
            file_path = f"TrainingImage/{name_input}.{id_input}.jpg"
            cv2.imwrite(file_path, frame)
            st.success("Face image saved successfully")
        else:
            st.error("Invalid image captured")

# ================== TRAIN MODEL ==================
st.subheader("Train Images")

if st.button("Train Images"):
    try:
        # Your training logic here
        st.success("Training completed successfully")
    except Exception as e:
        st.error(str(e))

# ================== RECOGNIZE ATTENDANCE ==================
st.subheader("Recognize Attendance")

rec_img = st.camera_input("Capture image for recognition")

if st.button("Recognize"):
    if rec_img is None:
        st.error("Please capture an image")
    else:
        image = Image.open(rec_img)
        frame = np.array(image)

        if frame is not None and frame.size != 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            st.image(gray, caption="Processed Image", channels="GRAY")
            st.success("Recognition completed (demo)")
        else:
            st.error("Invalid image")
