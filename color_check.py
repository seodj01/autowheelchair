import threading
import time
import cv2
import RPi.GPIO as GPIO
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import imutils

PWMA = 18
AIN1 = 22
AIN2 = 27

PWMB = 23
BIN1 = 25
BIN2 = 24

def motor_Back(speed):
    L_Motor.ChangeDutyCycle(speed)
    GPIO.output(AIN2,False)
    GPIO.output(AIN1,True)
    R_Motor.ChangeDutyCycle(speed)
    GPIO.output(BIN2,False)
    GPIO.output(BIN1,True)
    
def motor_go(speed):        
    L_Motor.ChangeDutyCycle(speed)
    GPIO.output(AIN2,True)
    GPIO.output(AIN1,False)
    R_Motor.ChangeDutyCycle(speed)
    GPIO.output(BIN2,True)
    GPIO.output(BIN1,False)

def motor_stop():
    L_Motor.ChangeDutyCycle(0)
    GPIO.output(AIN2,False)
    GPIO.output(AIN1,False)
    R_Motor.ChangeDutyCycle(0)
    GPIO.output(BIN2,False)
    GPIO.output(BIN1,False)
    
def motor_right(speed):
    L_Motor.ChangeDutyCycle(speed)
    GPIO.output(AIN2,True)
    GPIO.output(AIN1,False)
    R_Motor.ChangeDutyCycle(0)
    GPIO.output(BIN2,False)
    GPIO.output(BIN1,True)
    
def motor_left(speed):
    L_Motor.ChangeDutyCycle(0)
    GPIO.output(AIN2,False)
    GPIO.output(AIN1,True)
    R_Motor.ChangeDutyCycle(speed)
    GPIO.output(BIN2,True)
    GPIO.output(BIN1,False)
    
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(AIN2,GPIO.OUT)
GPIO.setup(AIN1,GPIO.OUT)
GPIO.setup(PWMA,GPIO.OUT)

GPIO.setup(BIN1, GPIO.OUT)
GPIO.setup(BIN2, GPIO.OUT)
GPIO.setup(PWMB, GPIO.OUT)

L_Motor = GPIO.PWM(PWMA,100)
L_Motor.start(0)

R_Motor = GPIO.PWM(PWMB,100)
R_Motor.start(0)

speedSet=20

def img_preprocess(image):
    # Resize the image to the size expected by the model (e.g., 66x200)
    processed_image = cv2.resize(image, (200, 66))
    
    # Convert the image to RGB (if needed)
    if len(processed_image.shape) == 2 or processed_image.shape[2] == 1:
        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2RGB)
    elif processed_image.shape[2] == 4:
        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGRA2RGB)
    elif processed_image.shape[2] == 3:
        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
    
    # Normalize the image (scale pixel values to the range [0, 1])
    processed_image = processed_image / 255.0
    
    # Add batch dimension
    processed_image = np.expand_dims(processed_image, axis=0)
    
    return processed_image

camera = cv2.VideoCapture(-1)
camera.set(3, 640)
camera.set(4, 480)

_, image = camera.read()
image_ok = 0
carState = "stop"
total_state = 0

MIN_CONTOUR_AREA = 10000

def track_blue_object(frame):
    global carState
    global total_state
    # Convert the frame from BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define the lower and upper bounds for the blue color in HSV
    lower_blue = np.array([36, 130, 46])
    upper_blue = np.array([113, 255, 255])

    # Create a mask for the blue color
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Apply morphological operations to the mask to reduce noise
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Find contours in the mask
    contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    max_contour = None
    max_contour_area = 0
    
    for c in contours:
        # Calculate the area of the contour
        contour_area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)

        if contour_area > MIN_CONTOUR_AREA and contour_area > max_contour_area and perimeter > 10:
            max_contour_area = contour_area
            max_contour = c
            
    
    if max_contour is not None:
        # Calculate the moments to find the centroid
        M = cv2.moments(max_contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
    
            # Draw a line at the centroid
            cv2.line(frame, (cx, 0), (cx, frame.shape[0]), (255, 0, 0), 1)
            cv2.line(frame, (0, cy), (frame.shape[1], cy), (255, 0, 0), 1)

            # Draw contours around the blue object
            cv2.drawContours(frame, [c], -1, (0, 255, 0), 1)

            print("Centroid X:", cx, "Centroid Y:", cy)
            if total_state == 1:
                if cy <= 150:
                    carState = "stop"
                else:
                    carState = "go"
            else:
                carState = "stop"
    else:
        if total_state == 1:
            carState = "go"

    return frame

def trace_thread():
    global image_ok
    global image
    
    while True:
        if image_ok == 1:
            frame = image

            # Call the function to track the blue object
            frame = track_blue_object(frame)

            cv2.imshow('frame', frame)
            image_ok = 0

def main():
    global carState
    global image_ok
    global image
    global total_state
    
    model_path = '/home/aicar/AI_CAR/lane_tarcing_model/lane_navigation_final.h5'
    model = load_model(model_path)
    
    try:
        while True:
            keyValue = cv2.waitKey(1)
        
            if keyValue == ord('q'):
                break
            elif keyValue == 82:
                print("go")
                carState = "go"
                total_state = 1
            elif keyValue == 84:
                print("stop")
                carState = "stop"
                total_state = 0
            
            image_ok = 0
            _, image = camera.read()
            image = cv2.flip(image, -1)
            image_ok = 1
            
            preprocessed = img_preprocess(image)
            cv2.imshow('pre', preprocessed[0])  # Display preprocessed image (remove batch dimension for display)
            
            X = np.asarray(preprocessed)
            steering_angle = int(model.predict(X)[0])
            print("predict angle:", steering_angle)
                
            if carState == "go":
                if steering_angle >= 73 and steering_angle <= 85:
                    print("go")
                    motor_go(speedSet)
                elif steering_angle > 85:
                    print("right")
                    motor_left(speedSet)
                elif steering_angle < 73:
                    print("left")
                    motor_right(speedSet)
            elif carState == "stop":
                motor_stop()
            
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    task1 = threading.Thread(target=trace_thread)
    task1.start()
    main()
    cv2.destroyAllWindows()
