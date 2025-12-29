from flask import Flask, render_template, request, jsonify
import os
from check_camera import camer
from Capture_Image import takeImages
from Train_Image import TrainImages
from Recognize import recognize_attendence

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/check_camera', methods=['POST'])
def check_camera():
    try:
        # Modify camer function to not show window, just check if camera works
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            cap.release()
            return jsonify({'status': 'success', 'message': 'Camera is working'})
        else:
            return jsonify({'status': 'error', 'message': 'Camera not accessible'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/capture_faces', methods=['POST'])
def capture_faces():
    data = request.get_json()
    Id = data.get('id')
    name = data.get('name')
    if not Id or not name:
        return jsonify({'status': 'error', 'message': 'ID and Name are required'})
    try:
        takeImages(Id, name)
        return jsonify({'status': 'success', 'message': f'Images captured for ID: {Id}, Name: {name}'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/train_images', methods=['POST'])
def train_images():
    try:
        TrainImages()
        return jsonify({'status': 'success', 'message': 'Images trained successfully'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/recognize', methods=['POST'])
def recognize():
    try:
        attendance_data = recognize_attendence()
        return jsonify({'status': 'success', 'attendance': attendance_data})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
