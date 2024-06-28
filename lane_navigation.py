import cv2 
import RPi.GPIO as GPIO  
import numpy as np
import torch
import torch.nn as nn
import torch.functional as F
from torchvision import transforms

PWMA = 18
AIN2 = 27
AIN1 = 22

PWMB = 23
BIN2 = 24
BIN1 = 25


GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(PWMA, GPIO.OUT)
GPIO.setup(AIN1, GPIO.OUT)
GPIO.setup(AIN2, GPIO.OUT)

GPIO.setup(PWMB, GPIO.OUT)
GPIO.setup(BIN1, GPIO.OUT)
GPIO.setup(BIN2, GPIO.OUT)

L_Motor = GPIO.PWM(PWMA, 500)
L_Motor.start(0)

R_Motor = GPIO.PWM(PWMB, 500)
R_Motor.start(0)

speedSet=30

def moving_Front(speed):        
    GPIO.output(AIN1,0)
    GPIO.output(AIN2,1)
    L_Motor.ChangeDutyCycle(speed)
    GPIO.output(BIN1,0)
    GPIO.output(BIN2,1)
    R_Motor.ChangeDutyCycle(speed)
    
def moving_Back(speed):
    GPIO.output(AIN1,1)
    GPIO.output(AIN2,0)
    L_Motor.ChangeDutyCycle(speed)
    GPIO.output(BIN1,1)
    GPIO.output(BIN2,0)
    R_Motor.ChangeDutyCycle(speed)

def moving_Left(speed):
    GPIO.output(AIN1,1)
    GPIO.output(AIN2,0)
    L_Motor.ChangeDutyCycle(speed)
    GPIO.output(BIN1,0)
    GPIO.output(BIN2,1)
    R_Motor.ChangeDutyCycle(speed)
    
def moving_Right(speed):
    GPIO.output(AIN1,0)
    GPIO.output(AIN2,1)
    L_Motor.ChangeDutyCycle(speed)
    GPIO.output(BIN1,1)
    GPIO.output(BIN2,0)
    R_Motor.ChangeDutyCycle(speed)

def moving_Stop():
    GPIO.output(AIN1,0)
    GPIO.output(AIN2,1)
    L_Motor.ChangeDutyCycle(0)
    GPIO.output(BIN1,0)
    GPIO.output(BIN2,1)
    R_Motor.ChangeDutyCycle(0)


class NvidiaModel(nn.Module):
    def __init__(self):
        super(NvidiaModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2, padding=2),
            nn.ELU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2, padding=2),
            nn.ELU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2, padding=2),
            nn.ELU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ELU()
        )
        self.flatten = nn.Flatten()
        self.fc_layers = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(4160, 100),
            nn.ELU(),
            nn.Linear(100, 50),
            nn.ELU(),
            nn.Linear(50, 10),
            nn.ELU(),
            nn.Linear(10, 1)  
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.fc_layers(x)
        x=180*torch.sigmoid(x)
        return x
    

def img_preprocess(image):
    height, _, _ = image.shape
    image = image[int(height/2):, :, :]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (200, 66))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image)
    return image.unsqueeze(0)  

def main():
    camera = cv2.VideoCapture(-1)
    camera.set(3, 640)
    camera.set(4, 480)
    model = NvidiaModel()
    model.load_state_dict(torch.load('/home/pi/AI_CAR/model/model_best_v5.pth', map_location=torch.device('cpu')))
    model.eval()

    carState = "stop"


    try:
        while True:
            keyValue=cv2.waitKey(1)

            if keyValue == ord('q'):
                break

            elif keyValue == 82:  # 'R' key for "go"
                print("go")
                carState = "go"

            elif keyValue == 84:  # 'T' key for "stop"
                print("stop")
                carState = "stop"
                
            _, image = camera.read()
            image = cv2.flip(image, -1)
            cv2.imshow('Original', image)

            preprocessed = img_preprocess(image)
            cv2.imshow('Preprocessed', preprocessed.numpy()[0].transpose(1, 2, 0)) 

            steering_angle = int(model(preprocessed).item())
            print("Predict angle:", steering_angle)

            if carState == "go":
                if steering_angle>=83 and steering_angle<=97:
                    print("Straight")
                    moving_Front(speedSet)

                elif steering_angle >97:
                    print("Right")
                    moving_Right(speedSet)

                elif steering_angle < 83:
                    print("Left")
                    moving_Left(speedSet)
            elif carState=="stop":
                moving_Stop()

        
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()
    cv2.destroyAllWindows()


