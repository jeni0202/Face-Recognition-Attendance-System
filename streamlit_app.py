import streamlit as st
import cv2
import os
from Capture_Image import takeImages
from Train_Image import TrainImages
from Recognize import recognize_attendence

# Create necessary directories
os.makedirs("TrainingImage", exist_ok=True)
os.makedirs("StudentDetails", exist_ok=True)
os.makedirs("Attendance", exist_ok=True)
os.makedirs("TrainingImageLabel", exist_ok=True)

st.title("Face Recognition Attendance System")

# Check Camera
if st.button("Check Camera"):
    st.info("Camera check is not available in the deployed web environment. Please run the application locally to test camera functionality.")
    
# Capture Faces
st.subheader("Capture Faces")
id_input = st.text_input("Enter ID")
name_input = st.text_input("Enter Name")
if st.button("Capture Faces"):
    if not id_input or not name_input:
        st.error("ID and Name are required")
    else:
        try:
            takeImages(id_input, name_input)
            st.success(f"Images captured for ID: {id_input}, Name: {name_input}")
        except Exception as e:
            st.error(f"Error: {str(e)}")

# Train Images
if st.button("Train Images"):
    try:
        TrainImages()
        st.success("Images trained successfully")
    except Exception as e:
        st.error(f"Error: {str(e)}")

# Recognize Attendance
if st.button("Recognize Attendance"):
    try:
        attendance_data = recognize_attendence()
        st.success("Attendance recognized")
        # Display attendance data if needed
    except FileNotFoundError as e:
        st.error("No student data found. Please capture faces first before recognizing attendance.")
    except Exception as e:
        st.error(f"Error: {str(e)}")
