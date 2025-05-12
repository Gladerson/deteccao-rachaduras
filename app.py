from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
from ultralytics import YOLO
import base64
from datetime import datetime

app = Flask(__name__)
model = YOLO("/home/gladersonjessika/Documentos/PPgTI/ComputerVision/Projeto/datasets/merged_crack_det.v1i.yolov11/detect_cracks/yolo11n_cracks/weights/last.pt")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    def generate_frames():
        camera = cv2.VideoCapture(1)
        while True:
            success, frame = camera.read()
            if not success:
                break
            results = model(frame)
            annotated_frame = results[0].plot()
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        camera.release()
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detect_image', methods=['POST'])
def detect_image():
    data = request.json
    image_data = data['image'].split(',')[1]
    img_bytes = base64.b64decode(image_data)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    results = model(img)
    classes = results[0].boxes.cls.tolist()
    is_crack_detected = 0 in classes
    # Annotar imagem
    result_img = results[0].plot()
    _, img_encoded = cv2.imencode('.jpg', result_img)
    img_base64 = base64.b64encode(img_encoded).decode('utf-8')
    return jsonify({'crack_detected': is_crack_detected, 'annotated_image': img_base64})

@app.route('/detect_time', methods=['GET'])
def detect_time():
    camera = cv2.VideoCapture(1)
    success, frame = camera.read()
    camera.release()
    if not success:
        return jsonify({'crack_detected': False})
    results = model(frame)
    classes = results[0].boxes.cls.tolist()
    is_crack_detected = 0 in classes
    timestamp = datetime.now().strftime('%H:%M:%S') if is_crack_detected else ''
    return jsonify({'crack_detected': is_crack_detected, 'timestamp': timestamp})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
