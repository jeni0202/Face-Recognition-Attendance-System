import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import os

# Create necessary directories
os.makedirs("TrainingImage", exist_ok=True)
os.makedirs("StudentDetails", exist_ok=True)
os.makedirs("Attendance", exist_ok=True)
os.makedirs("TrainingImageLabel", exist_ok=True)

from Capture_Image import takeImages
from Train_Image import TrainImages
from Recognize import recognize_attendence

st.title("Face Recognition Attendance System")

# Check Camera
if st.button("Check Camera"):
    try:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            
            st.success("Camera is working")
        else:
            st.error("Camera not accessible")
    except Exception as e:
        st.error(f"Error: {str(e)}")

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

class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        try:
            processed_frame, attendance = recognize_attendence(img)
            return processed_frame
        except Exception as e:
            # If recognition fails, return original frame
            return img

# Recognize Attendance
st.subheader("Recognize Attendance")

# Session state to control starting/stopping recognition (webcam)
if "recognizing" not in st.session_state:
    st.session_state["recognizing"] = False

col1, col2 = st.columns(2)
with col1:
    if st.button("Start Recognize"):
        st.session_state["recognizing"] = True
with col2:
    if st.button("Stop Recognize"):
        st.session_state["recognizing"] = False

if st.session_state["recognizing"]:
    st.info("Webcam active. Allow camera access in your browser if prompted.")
    webrtc_streamer(key="recognize-stream", video_transformer_factory=VideoTransformer)
else:
    st.warning("Click 'Start Recognize' to open the webcam and begin face recognition.")

# Optionally show the most recent attendance file (if any)
latest_csv = None
if os.path.isdir("Attendance"):
    try:
        files = [f for f in os.listdir("Attendance") if f.lower().endswith(".csv")]
        if files:
            files.sort(key=lambda f: os.path.getmtime(os.path.join("Attendance", f)), reverse=True)
            latest_csv = os.path.join("Attendance", files[0])
    except Exception:
        latest_csv = None

if latest_csv:
    st.caption("Latest attendance record:")
    try:
        import pandas as pd
        df_latest = pd.read_csv(latest_csv)
        if not df_latest.empty:
            st.dataframe(df_latest, use_container_width=True)
        else:
            st.write("Latest attendance file is empty.")
    except Exception as e:
        st.write(f"Could not read latest attendance file: {e}")
        
