import threading
import time
import cv2
import RPi.GPIO as GPIO
import numpy as np
from tflite_runtime.interpreter import Interpreter
from tensorflow.keras.models import load_model
from flask import Flask, Response, render_template, jsonify

app = Flask(__name__)

# Motor pins
PWMA = 18
AIN1 = 22
AIN2 = 27
PWMB = 23
BIN1 = 25
BIN2 = 24

log_data = []
predicted_angle_log = []
stop_flag = threading.Event()

# Motor functions
def motor_Back(speed):
    L_Motor.ChangeDutyCycle(speed)
    GPIO.output(AIN1, 1)
    GPIO.output(AIN2, 0)
    R_Motor.ChangeDutyCycle(speed)
    GPIO.output(BIN1, 1)
    GPIO.output(BIN2, 0)

def motor_go(speed):
    L_Motor.ChangeDutyCycle(speed)
    GPIO.output(AIN2, True)
    GPIO.output(AIN1, False)
    R_Motor.ChangeDutyCycle(speed)
    GPIO.output(BIN2, True)
    GPIO.output(BIN1, False)

def motor_Stop():
    L_Motor.ChangeDutyCycle(0)
    GPIO.output(AIN2, False)
    GPIO.output(AIN1, False)
    R_Motor.ChangeDutyCycle(0)
    GPIO.output(BIN2, False)
    GPIO.output(BIN1, False)

def motor_Right(speed):
    L_Motor.ChangeDutyCycle(speed)
    GPIO.output(AIN2, True)
    GPIO.output(AIN1, False)
    R_Motor.ChangeDutyCycle(speed)
    GPIO.output(BIN2, False)
    GPIO.output(BIN1, True)

def motor_Left(speed):
    L_Motor.ChangeDutyCycle(speed)
    GPIO.output(AIN2, False)
    GPIO.output(AIN1, True)
    R_Motor.ChangeDutyCycle(speed)
    GPIO.output(BIN2, True)
    GPIO.output(BIN1, False)

# GPIO setup
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(AIN2, GPIO.OUT)
GPIO.setup(AIN1, GPIO.OUT)
GPIO.setup(PWMA, GPIO.OUT)
GPIO.setup(BIN1, GPIO.OUT)
GPIO.setup(BIN2, GPIO.OUT)
GPIO.setup(PWMB, GPIO.OUT)

L_Motor = GPIO.PWM(PWMA, 100)
L_Motor.start(0)
R_Motor = GPIO.PWM(PWMB, 100)
R_Motor.start(0)

speedSet = 5

def load_labels(label_file):
    with open(label_file, 'r') as f:
        lines = f.readlines()
    labels = {}
    current_index = 0
    for i, line in enumerate(lines):
        label = line.strip()
        if label != '???':  # '???'를 제외하고 매핑
            labels[current_index] = label
            current_index += 1
    return labels

label_file = "/home/aicar/AI_CAR/OpencvDnn/models/labelmap.txt"
classNames = load_labels(label_file)

def id_class_name(class_id, classes):
    return classes.get(class_id, 'Unknown')

def img_preprocess(image):
    height, _, _ = image.shape
    image = image[int(height/2):,:,:]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    image = cv2.GaussianBlur(image, (3,3), 0)
    image = cv2.resize(image, (200,66))
    image = image / 255
    return image

camera = cv2.VideoCapture(-1)
camera.set(3, 640)
camera.set(4, 480)

_, image = camera.read()
image_ok = 0

box_size = 0
carState = "stop"

object_detected = False
frame_count = 0

# TensorFlow Lite 모델 로드
interpreter = Interpreter(model_path="/home/aicar/AI_CAR/OpencvDnn/models/detect.tflite")
interpreter.allocate_tensors()

# 입력 및 출력 텐서 정보 가져오기
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def obdect_detection_thread():
    global image
    global image_ok
    global carState
    global object_detected
    global frame_count

    while not stop_flag.is_set():
        if image_ok == 1:
            frame_count += 1
            if frame_count % 5 != 0:  # 매 3번째 프레임만 처리
                continue

            imagednn = image
            image_height, image_width, _ = imagednn.shape

            input_data = cv2.resize(imagednn, (300, 300))
            input_data = np.expand_dims(input_data, axis=0)
            input_data = input_data.astype(np.uint8)

            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()

            detection_boxes = interpreter.get_tensor(output_details[0]['index'])
            detection_classes = interpreter.get_tensor(output_details[1]['index'])
            detection_scores = interpreter.get_tensor(output_details[2]['index'])
            num_detections = interpreter.get_tensor(output_details[3]['index'])

            detection_class_names = [id_class_name(int(cls), classNames) for cls in detection_classes[0]]

            object_detected = False

            for i in range(int(num_detections[0])):
                class_id = int(detection_classes[0][i])
                score = detection_scores[0][i]
                if score > 0.5:
                    class_name = id_class_name(class_id, classNames)
                    log_entry = f"Detected: {class_name} with confidence {score:.2f}"
                    log_data.append(log_entry)  # 로그 데이터를 log_data에 추가


                    if class_name in ["person", "stop sign", "bicycle", "car", "motorcycle", "bus"]:
                        box_x = detection_boxes[0][i][1] * image_width
                        box_y = detection_boxes[0][i][0] * image_height
                        box_width = detection_boxes[0][i][3] * image_width
                        box_height = detection_boxes[0][i][2] * image_height

                        carState = "stop"
                        object_detected = True
                        motor_Stop()

                        cv2.rectangle(image, (int(box_x), int(box_y)), (int(box_width), int(box_height)), (23, 230, 210), thickness=2)
                        text = f"{class_name}: {score:.2f}"
                        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                        cv2.rectangle(image, (int(box_x), int(box_y) - 20), (int(box_x) + w, int(box_y)), (23, 230, 210), -1)
                        cv2.putText(image, text, (int(box_x), int(box_y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

            if not object_detected and carState == "stop":
                carState = "go"

            image_ok = 0


def lane_detection_thread():
    global image
    global image_ok
    global carState
    global frame_count

    model_path = '/home/aicar/AI_CAR/new_lane_model/ss0120.h5'
    lane_model = load_model(model_path)

    carState = "stop"

    while not stop_flag.is_set():
        frame_count += 1
        if frame_count % 5 != 0:  # 매 3번째 프레임만 처리
            continue

        image_ok = 0
        ret, image = camera.read()
        if not ret:
            continue
        image = cv2.flip(image, 0)  # 상하 반전
        image_ok = 1

        preprocessed = img_preprocess(image)

        X = np.asarray([preprocessed])
        steering_angle = int(lane_model.predict(X)[0])
        predicted_angle_log.append(f"Predicted angle: {steering_angle}")

        if carState == "go":
            if 60 <= steering_angle <= 110:
                motor_go(speedSet)
            elif steering_angle > 110:
                motor_Right(speedSet)
            elif steering_angle < 60:
                motor_Left(speedSet)

        elif carState == "stop":
            motor_Stop()


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    def generate():
        global image
        while True:
            ret, frame = camera.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 0)  # 상하 반전
            ret, jpeg = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            frame = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start')
def start():
    global carState
    stop_flag.clear()  # Stop flag 초기화
    carState = "go"
    return "Car started"

@app.route('/stop')
def stop():
    global carState
    carState = "stop"
    motor_Stop()
    stop_flag.set()  # Stop flag 설정
    return "Car stopped"

@app.route('/logs')
def logs():
    global log_data
    global predicted_angle_log
    return jsonify({"logs": log_data, "predicted_angles": predicted_angle_log})


if __name__ == '__main__':
    detection_thread = threading.Thread(target=obdect_detection_thread)
    lane_thread = threading.Thread(target=lane_detection_thread)
    detection_thread.start()
    lane_thread.start()
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
    detection_thread.join()
    lane_thread.join()
    GPIO.cleanup()
