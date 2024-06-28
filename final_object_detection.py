import threading
import time
import cv2
import RPi.GPIO as GPIO  
import numpy as np
from tflite_runtime.interpreter import Interpreter
from tensorflow.keras.models import load_model

PWMA = 18
AIN1 = 22
AIN2 = 27

PWMB = 23
BIN1 = 25
BIN2 = 24

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

speedSet = 30

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
camera.set(3,640)
camera.set(4,480)

_, image = camera.read()
image_ok = 0

box_size = 0
carState = "stop"

object_detected = False

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

    while True:
        if image_ok == 1:
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

            print(f"Num detections: {num_detections}")
            print(f"Detection boxes: {detection_boxes}")
            print(f"Detection classes: {detection_classes}")
            print(f"Detection scores: {detection_scores}")

            detection_class_names = [id_class_name(int(cls), classNames) for cls in detection_classes[0]]
            print(f"Detection class names: {detection_class_names}")

            object_detected = False

            for i in range(int(num_detections[0])):
                class_id = int(detection_classes[0][i])
                score = detection_scores[0][i]
                if score > 0.5:
                    class_name = id_class_name(class_id, classNames)
                    print(f"Detected: {class_name} with confidence {score}")
                    if class_name in ["person","traffic light", "stop sign", "bicycle", "car", "motorcycle", "bus"]:
                        box_x = detection_boxes[0][i][1] * image_width
                        box_y = detection_boxes[0][i][0] * image_height
                        box_width = detection_boxes[0][i][3] * image_width
                        box_height = detection_boxes[0][i][2] * image_height

                        carState = "stop"
                        object_detected = True
                        print("auto stop")
                        motor_Stop()

                        cv2.rectangle(image, (int(box_x), int(box_y)), (int(box_width), int(box_height)), (23, 230, 210), thickness=2)
                        text = f"{class_name}: {score:.2f}"
                        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                        cv2.rectangle(image, (int(box_x), int(box_y) - 20), (int(box_x) + w, int(box_y)), (23, 230, 210), -1)
                        cv2.putText(image, text, (int(box_x), int(box_y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

            if not object_detected and carState == "stop":
                carState = "go"
                print("object not detected, resuming movement")

            cv2.imshow('Detection', image)  # 객체 감지 결과를 화면에 표시
            image_ok = 0




def main():
    global image
    global image_ok
    global carState

    model_path = '/home/aicar/AI_CAR/new_lane_model/ss0120.h5'
    lane_model = load_model(model_path)

    carState = "stop"  # 초기 상태를 "stop"으로 설정

    try:
        while True:
            keyValue = cv2.waitKey(1)

            if keyValue == ord('q'):
                break

            elif keyValue == 82:  # '방향키 위' 버튼을 누르면 "go" 상태로 변경
                print("go")
                carState = "go"

            elif keyValue == 84:  # '방향키 아래' 버튼을 누르면 "stop" 상태로 변경
                print("stop")
                carState = "stop"
            
            _, image = camera.read()
            image = cv2.flip(image, -1)
            image_ok = 1  # 프레임이 준비되었음을 표시

            preprocessed = img_preprocess(image)
            cv2.imshow('pre', preprocessed)

            input_data = np.expand_dims(preprocessed, axis=0).astype(np.float32)

            steering_angle = lane_model.predict(input_data)[0]
            print("Predict angle:", steering_angle)

            if carState == "go":
                if steering_angle >= 60 and steering_angle <= 110:
                    print("Straight")
                    motor_go(speedSet)
                elif steering_angle > 110:
                    print("right")
                    motor_Right(speedSet)
                elif steering_angle < 60:
                    print("left")
                    motor_Left(speedSet)

            elif carState == "stop":
                motor_Stop()

    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    task1 = threading.Thread(target=obdect_detection_thread)
    task1.start()
    main()
    cv2.destroyAllWindows()
    GPIO.cleanup()


if __name__ == '__main__':
    task1 = threading.Thread(target=obdect_detection_thread)
    task1.start()
    main()
    cv2.destroyAllWindows()
    GPIO.cleanup()
 