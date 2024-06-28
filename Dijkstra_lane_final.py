import threading
import time
import cv2
import RPi.GPIO as GPIO
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import imutils
import heapq
import matplotlib
import matplotlib.pyplot as plt

#matplotlib.use('PS')

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
    GPIO.output(AIN2,0)
    L_Motor.ChangeDutyCycle(0)
    GPIO.output(BIN1,0)
    GPIO.output(BIN2,0)

#----------------------------------------------------------------
matrix_map = [
    [6, 4, 1],
    [2, 3, 8],
    [7, 4, 1]
]
start_coord = (0, 0)
end_coord = (2, 2)
path = []
steering_ok=True
map_go=True

def dijkstra(matrix, start, end):
    # (previous node, current node) for path tracking
    prev_nodes = [[None] * len(matrix[0]) for _ in range(len(matrix))]
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    # Initializing tables for shortest path
    distance = [[float('inf')] * len(matrix[0]) for _ in range(len(matrix))]
    distance[start[0]][start[1]] = matrix[start[0]][start[1]]

    min_heap = [(matrix[start[0]][start[1]], start)]

    while min_heap:
        current_dist, current_node = heapq.heappop(min_heap)

        for i, (dr, dc) in enumerate(directions):
            new_row, new_col = current_node[0] + dr, current_node[1] + dc

            if 0 <= new_row < len(matrix) and 0 <= new_col < len(matrix[0]):
                # If direction changes, add 1 to the weight
                weight_change = 1 if i != matrix[current_node[0]][current_node[1]] else 0
                new_dist = current_dist + matrix[new_row][new_col] + weight_change

                if new_dist < distance[new_row][new_col]:
                    distance[new_row][new_col] = new_dist
                    prev_nodes[new_row][new_col] = current_node
                    heapq.heappush(min_heap, (new_dist, (new_row, new_col)))

    path = []
    current = end
    while current is not None:
        path.append(current)
        current = prev_nodes[current[0]][current[1]]

    return distance, path


        

def update_plot():
    global current_slope, current_position, previous_positions
    if previous_positions:
        previous_positions_np = np.array(previous_positions)
        plt.plot(previous_positions_np[:, 0], previous_positions_np[:, 1], 'g-')

    plt.scatter(current_position[0], current_position[1], c='r')
    
    plt.pause(0.01)
    

def move_along_path():
    global x_pos, y_pos, steering_angle, steering_ok, map_go
    rrrr, patht = dijkstra(matrix_map, start_coord, end_coord)
    varr=0
    time.sleep(0.5)
    directions = []
    
    for i in range(len(patht) - 1):
        current_node = patht[i+1]
        next_node = patht[i]
        direction = (next_node[0] - current_node[0], next_node[1] - current_node[1])
        directions.append(direction)
    print('patht : ',patht)
    print('directions : ',directions)
    print('patht.0 0:',patht[(len(directions)-1)][0])
    print('patht.0 1:',patht[(len(directions)-1)][1])

    while True:

        
        for xy in range(len(directions)-1):
            varr=0
            #print("True False:",(y_pos==patht[xy][0]) and (x_pos==patht[xy][1]))
            #if (abs(y_pos - patht[xy][0]) < 0.5) and (abs(x_pos - patht[xy][1]) < 0.5):
            if (x_pos == int(patht[xy+1][1]))  and (y_pos == int(patht[xy+1][0])):
                print('succeed')
                steering_ok=False
                                  
                fre_dire=directions[xy+1]
                cur_dire=directions[xy]
                print('fre_dire: ',fre_dire,'cur_dire: ',cur_dire)
                                   
                x_dif=cur_dire[0]-fre_dire[0]
                y_dif=cur_dire[1]-fre_dire[1]
                print('x_dif: ',x_dif,'y_dif: ',y_dif)
                
                map_go=False
                
                if x_dif<1 and y_dif<1:
                    #steering_ok=True
                    steering_angle=90
                    print('change go')
                    map_go=True
                    time.sleep(7)
                else:                       
                    time.sleep(4.5)
                map_go=True
                
                if x_dif>0:
                    steering_angle=45                  
                    print('change left')
                    time.sleep(3)
                    
                elif y_dif>0:
                    steering_angle=135                  
                    print('change right')
                    time.sleep(3)
                x_pos=100
                y_pos=100
                varr=1
        if varr>0:          
            x_pos=100
            y_pos=100
            varr=0
        steering_ok=True
            
            
                                                                       
def shortest_path_and_mapping():
    global matrix_map, start_coord, end_coord, path
    global current_slope, current_position, previous_positions
    global carState, steering_angle, map_go
    
    plt.figure()
    plt.axis('equal')
    plt.show(block=False)
    result, path = dijkstra(matrix_map, start_coord, end_coord)
    if result[end_coord[0]][end_coord[1]] != float('inf'):
        
        path_x, path_y = zip(*path)
        plt.imshow(np.ones_like(matrix_map), cmap='gray', alpha=0)
        
        for i in range(len(path) - 1):
            x_values = [path[i][1], path[i + 1][1]]
            y_values = [path[i][0], path[i + 1][0]]
            plt.plot(x_values, y_values, color='blue', linestyle='dashed')
            weight_label = f"{matrix_map[path[i][0]][path[i][1]]}"
            plt.text((x_values[0] + x_values[1]) / 2, (y_values[0] + y_values[1]) / 2, weight_label, color='blue',
                     ha='center', va='center')
            
        plt.gca().invert_yaxis()
        # Add horizontal and vertical lines at 0.5 and 1.5
        plt.axhline(y=0, color='black', linestyle='dashed', linewidth=1)
        plt.axhline(y=1, color='black', linestyle='dashed', linewidth=1)
        plt.axhline(y=2, color='black', linestyle='dashed', linewidth=1)
        plt.axvline(x=0, color='black', linestyle='dashed', linewidth=1)
        plt.axvline(x=1, color='black', linestyle='dashed', linewidth=1)
        plt.axvline(x=2, color='black', linestyle='dashed', linewidth=1)
        
        plt.scatter(path_y, path_x, color='red', marker='x', label='Shortest path')
        plt.scatter(start_coord[1], start_coord[0], color='green', marker='o', label='Start point')
        plt.scatter(end_coord[1], end_coord[0], color='blue', marker='o', label='End point')

        
        for i in range(len(matrix_map)):
            for j in range(len(matrix_map[0])):
                plt.text(j, i, str(matrix_map[i][j]), color='black', ha='center', va='center')

        plt.legend()
    while True:
                
        previous_positions.append(current_position.copy())
        
        if carState == "go":
            if steering_angle >= 85 and steering_angle <= 95 and map_go==True:
                current_position += 0.2*np.array([np.cos(current_slope), np.sin(current_slope)], dtype=np.float64)
                
            elif steering_angle > 96:
                current_slope -= np.radians(7)
                #current_position += 0.01*np.array([np.cos(current_slope), np.sin(current_slope)], dtype=np.float64)
                
            elif steering_angle < 84:
                current_slope += np.radians(7)
                #current_position += 0.01*np.array([np.cos(current_slope), np.sin(current_slope)], dtype=np.float64)
        
           
        update_plot()


camera = cv2.VideoCapture(-1)
camera.set(3, 640)
camera.set(4, 480)

_, image = camera.read()
image_ok = 0
carState = "stop"
total_state=0

MIN_CONTOUR_AREA = 5000

color=5
#0,1:red 2:green 3:blue
shape=0
#3,4,5
pre_color=5
pre_shape=0

current_position = np.array([0.0, 0.0], dtype=np.float64)
current_slope = 0
previous_positions = []
x_pos=100
y_pos=100


def track_color_object(frame):
    global carState
    global color, shape, pre_color, pre_shape
    global x_pos, y_pos
    global current_slope, current_position, previous_positions
    global matrix_map
    global steering_angle, steering_ok  # 추가

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    lower_color = np.array([
        [0, 76, 76],     # Red (low)
        [170, 76, 76],   # Red (high)
        [78, 200, 100],  # Green (low)
        [98, 140, 140]   # Blue (low)
    ])

    upper_color = np.array([
        [10, 255, 255],  # Red (low)
        [180, 255, 255], # Red (high)
        [88, 255, 148],  # Green (high)
        [108, 180, 163]  # Blue (high)
    ])

    max_contour = None
    max_contour_area = 0
    
    for i in range(4):
        mask = cv2.inRange(hsv, lower_color[i], upper_color[i])
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for c in contours:
            epsilon = 0.06 * cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, epsilon, True)
            vertices = len(approx)
            
            contour_area = cv2.contourArea(c)
            perimeter = cv2.arcLength(c, True)

            if contour_area > MIN_CONTOUR_AREA and contour_area > max_contour_area and perimeter > 10:
                if 3 <= vertices <= 5:
                    max_contour_area = contour_area
                    max_contour = c
                    vernum = vertices
                    color_num = i
                
    if max_contour is not None:
        M = cv2.moments(max_contour)
        
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            
            if cx > 50 and cx < 350 and cy > 50 and cy < 400:
                cv2.line(frame, (cx, 0), (cx, frame.shape[0]), (255, 0, 0), 1)
                cv2.line(frame, (0, cy), (frame.shape[1], cy), (255, 0, 0), 1)
                cv2.drawContours(frame, [c], -1, (0, 255, 0), 1)
                
                shape = vernum
                color = color_num

                print("Centroid X:", cx, "Centroid Y:", cy)
                print("color:", color, "shape:", shape)


    if pre_color != color or pre_shape != shape:
        if color in [0, 1]:
            x_pos = 0
        elif color == 2:
            x_pos = 1
        elif color == 3:
            x_pos = 2
        pre_color = color
            
        if shape == 3:
            y_pos = 0
        elif shape == 4:
            y_pos = 1
        elif shape == 5:
            y_pos = 2
        pre_shape = shape
        print('x:', x_pos, 'y:', y_pos)
        current_position = np.array([x_pos, y_pos], dtype=np.float64)


def trace_thread():
    global image_ok
    global image
      
    while True:
        if image_ok==1:
            frame = image
            
            pts1 = np.float32([[220,250],[432,250],[30,440],[620,440]])
            pts2 = np.float32([[0,0],[300,0],[0,450],[300,450]])
            M = cv2.getPerspectiveTransform(pts1,pts2)
            
            flatframe = cv2.warpPerspective(frame, M, (300,450))
            # Call the function to track the blue object
            track_color_object(flatframe)

            cv2.imshow('frame', flatframe)
            image_ok=0

        
def img_preprocess(image):
    height, _, _=image.shape
    image=image[int(height/2):,:,:]
    image=cv2.cvtColor(image,cv2.COLOR_BGR2YUV)
    image=cv2.GaussianBlur(image,(3,3),0)
    image=cv2.resize(image,(200,66))  
    image=image/255
    return image

carState = "stop"
steering_angle = 0

def main():
    global carState, steering_angle, steering_ok
    global current_slope, current_position, previous_positions
    global image_ok
    global image
    
    model_path = '/home/aicar/AI_CAR/new_lane_model/ss0120.h5'
    model = load_model(model_path)
    
    try:
        while True:
            keyValue = cv2.waitKey(1)
        
            if keyValue == ord('q') :
                break
            elif keyValue == 82 :
                print("go")
                carState = "go"
            elif keyValue == 84 :
                print("stop")
                carState = "stop"
            
            image_ok = 0
            _, image = camera.read()
            image = cv2.flip(image, -1)
            image_ok = 1

            # 이미지 전처리 및 모델 예측
            preprocessed = img_preprocess(image)
            cv2.imshow('Preprocessed', preprocessed)

            X = np.asarray([preprocessed])
            if steering_ok == True:
                # 모델 예측 결과를 사용하여 steering_angle 설정
                steering_angle = int(model.predict(X)[0])
                print("predict angle:", steering_angle)

            # 로봇 상태에 따른 이동
            if carState == "go":
                if steering_angle >= 75 and steering_angle <= 100:
                    print("Straight")
                    moving_Front(speedSet)
                elif steering_angle > 100:
                    print("right")
                    moving_Right(speedSet)
                elif steering_angle < 75:
                    print("left")
                    moving_Left(speedSet)
            elif carState == "stop":
                moving_Stop()

    except KeyboardInterrupt:
        pass
    finally:
        moving_Stop()
        GPIO.cleanup()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    # 도형 인식 스레드 시작
    task1 = threading.Thread(target=trace_thread)
    task1.start()
    # 최단 경로 및 매핑 스레드 시작
    task2 = threading.Thread(target=shortest_path_and_mapping)
    task2.start() 
    # 경로를 따라 이동하는 스레드 시작
    task3 = threading.Thread(target=move_along_path)
    task3.start()
    # 메인 함수 실행
    main()

    
