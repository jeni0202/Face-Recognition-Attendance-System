def camera_streamlit():
    import streamlit as st
    import cv2
    import numpy as np
    from PIL import Image

    st.subheader("Camera Check (Streamlit)")

    img = st.camera_input("Capture image to detect face")

    if img is not None:
        image = Image.open(img)
        frame = np.array(image)

        if frame is None or frame.size == 0:
            st.error("Invalid image")
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        face_cascade = cv2.CascadeClassifier(
            "haarcascade_frontalface_default.xml"
        )

        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30)
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(
                frame, (x, y), (x + w, y + h), (10, 159, 255), 2
            )

        st.image(frame, caption="Face Detection Result", channels="BGR")

        if len(faces) > 0:
            st.success(f"Face detected: {len(faces)}")
        else:
            st.warning("No face detected")
