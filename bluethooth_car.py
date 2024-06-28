import serial #시리얼 사용
import threading  #쓰레드기능을 사용하기 위해 사용
import time #time기능을 위해 사용
import RPi.GPIO as GPIO  #GPIO핀을 사용하기 위해 사용

BlSerial = serial.Serial("/dev/ttyS0", baudrate=9600, timeout=1.0) #dev/ttyS0을 시리얼통신핀으로 사용한다.
gData = ""  #gData의 변수를 생성하고 빈 문자열로 초기화

PWMA = 18
AIN2 = 27
AIN1 = 22

PWMB = 23
BIN2 = 24
BIN1 = 25

SW_F = 5
SW_B = 19
SW_R = 6
SW_L = 13

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(PWMA, GPIO.OUT)
GPIO.setup(AIN1, GPIO.OUT)
GPIO.setup(AIN2, GPIO.OUT)
GPIO.setup(PWMB, GPIO.OUT)
GPIO.setup(BIN1, GPIO.OUT)
GPIO.setup(BIN2, GPIO.OUT)
GPIO.setup(SW_R,GPIO.IN,pull_up_down=GPIO.PUD_DOWN)
GPIO.setup(SW_F,GPIO.IN,pull_up_down=GPIO.PUD_DOWN)
GPIO.setup(SW_B,GPIO.IN,pull_up_down=GPIO.PUD_DOWN)
GPIO.setup(SW_L,GPIO.IN,pull_up_down=GPIO.PUD_DOWN)


L_Motor = GPIO.PWM(PWMA, 500)
L_Motor.start(0)

R_Motor = GPIO.PWM(PWMB, 500)
R_Motor.start(0)

#LED들의 이름을 지어 핀들을 연결해주었다.
LED_FR = 16
LED_FL = 26 
LED_BR = 21
LED_BL = 20

#사용할 LED를 출력설정을 해준다.
GPIO.setup(LED_FR, GPIO.OUT)
GPIO.setup(LED_FL, GPIO.OUT)
GPIO.setup(LED_BR, GPIO.OUT)
GPIO.setup(LED_BL, GPIO.OUT)


#앞,뒤,옆으로 움직이는 부분, 그리고 LED의 동작을 사용하기 쉽게 함수로 선언
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
    
def OUT_LED_GO():
    GPIO.output(LED_FR,GPIO.HIGH)
    GPIO.output(LED_FL,GPIO.HIGH)
    GPIO.output(LED_BR,GPIO.LOW)
    GPIO.output(LED_BL,GPIO.LOW)
    
def OUT_LED_BACK():
    GPIO.output(LED_FR,GPIO.LOW)
    GPIO.output(LED_FL,GPIO.LOW)
    GPIO.output(LED_BR,GPIO.HIGH)
    GPIO.output(LED_BL,GPIO.HIGH)

def OUT_LED_LEFT():
    GPIO.output(LED_FR,GPIO.LOW)
    GPIO.output(LED_FL,GPIO.HIGH)
    GPIO.output(LED_BR,GPIO.LOW)
    GPIO.output(LED_BL,GPIO.HIGH)

def OUT_LED_RIGHT():
    GPIO.output(LED_FR,GPIO.HIGH)
    GPIO.output(LED_FL,GPIO.LOW)
    GPIO.output(LED_BR,GPIO.HIGH)
    GPIO.output(LED_BL,GPIO.LOW)

def OUT_LED_STOP():
    GPIO.output(LED_FR,GPIO.LOW)
    GPIO.output(LED_FL,GPIO.LOW)
    GPIO.output(LED_BR,GPIO.LOW)
    GPIO.output(LED_BL,GPIO.LOW)

def serial_thread():  #시리얼 통신 스레드
        global gData  #serial_tread라는 함수안에서 gData를 사용하기위해 선언해준다.
        while True:
            data = BlSerial.readline() #한줄씩 값을 받는다. 
            data = data.decode() #decode로 시리얼통신의 bytes 타입을 문자열 타입으로 변경한다.
            gData = data  #받은 데이터를 gData에 대입해준다.
    
def main():
    global gData  #gData를 사용하기 위해 선언해준다.
    try:
        while True:
            if gData.find("go") >= 0:  #find를 통해서 go라는 값을 찾는다. 찾는다면 조건을 실행
                gData = ""    #gData를 비어둔다. 비어두지 않으면 gData는 계속 go로 항상 참
                print("ok go")   #go가 들어온것을 확인하기 위해서 ok go라는 대사를 출력
                moving_Front(50) 
                OUT_LED_GO() 
                
            elif gData.find("back") >= 0: 
                gData = ""
                print("ok back")
                moving_Back(50)
                OUT_LED_BACK()
            elif gData.find("left") >= 0:
                gData = ""
                print("ok left")
                moving_Left(50)
                OUT_LED_LEFT()
            elif gData.find("right") >= 0:
                gData = ""
                print("ok right")
                moving_Right(50)
                OUT_LED_RIGHT()
            elif gData.find("stop") >= 0:
                gData = ""
                print("ok stop")
                moving_Stop()
                OUT_LED_STOP()
            if GPIO.input(SW_F) == 1 :    #비상멈춤 버튼이다. Front스위치를 누르면 멈추게한다. 
                moving_Stop()
                OUT_LED_STOP()
                
                      
    except KeyboardInterrupt:  #Ctrl+C 사용 시 종료한다. 
        pass

if __name__ == '__main__': #스레드 실행과 종료 처리
    task1 = threading.Thread(target = serial_thread)   #task1을  이름으로 쓰레드를 생성한다. 
    task1.start()   #쓰레드를 시작한다. 
    main()  #main함수를 실행한다.
    BlSerial.close() #main함수가 끝나면 반환한다.
    GPIO.cleanup()