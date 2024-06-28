import cv2
import RPi.GPIO as GPIO
import time

PWMA = 18
AIN2 = 27
AIN1 = 22

PWMB = 23
BIN2 = 24
BIN1 = 25

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

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)

GPIO.setup(PWMA, GPIO.OUT)
GPIO.setup(AIN1, GPIO.OUT)
GPIO.setup(AIN2, GPIO.OUT)

GPIO.setup(PWMB, GPIO.OUT)
GPIO.setup(BIN1, GPIO.OUT)
GPIO.setup(BIN2, GPIO.OUT)

L_Motor = GPIO.PWM(PWMA, 100)
L_Motor.start(0)

R_Motor = GPIO.PWM(PWMB, 100)
R_Motor.start(0)

speedSet=30

def main():
    camera=cv2.VideoCapture(-1)
    camera.set(3,640)
    camera.set(4,480)

    filepath="/home/aicar/project/bin/video/"
    i=0
    carState="stop"
    
    while(camera.isOpened()):
        keyValue=cv2.waitKey(10)


        if keyValue == ord('q'):
            break
        
        elif keyValue ==82:
            print("go")
            carState="go"
            moving_Front(speedSet)

        elif keyValue==84:
            print("stop")
            carState="stop"
            moving_Stop()

        elif keyValue==81:
            print("left")
            carState="left"
            moving_Left(speedSet)

        elif keyValue==83:
            print("right")
            carState="right"
            moving_Right(speedSet)
                
        _, image=camera.read()
        image=cv2.flip(image,-1)
        cv2.imshow('Original',image)
            
        height, _, _=image.shape
        save_image=image[int(height/2):,:,:]
        save_image=cv2.cvtColor(save_image,cv2.COLOR_BGR2YUV)
        save_image=cv2.GaussianBlur(save_image,(3,3),0)
        save_image=cv2.resize(save_image,(200,66))
        cv2.imshow('Save',save_image)

        
            
        if carState=="left":
            cv2.imwrite("%s_%05d_%03d.png" % (filepath,i,45),image)
            i+=1
        elif carState=="right":
            cv2.imwrite("%s_%05d_%03d.png" % (filepath,i,135),image)
            i+=1
        elif carState=="go":
            cv2.imwrite("%s_%05d_%03d.png" % (filepath,i,90),image)
            i+=1
        
    cv2.destoryAllWindows()
        
if __name__== '__main__':
    main()  
    GPIO.cleanup()
