import cv2
import mediapipe as mp
import pygame
from math import *

WINDOW_SIZE =  800
ROTATE_SPEED = 0.02
window = pygame.display.set_mode( (WINDOW_SIZE, WINDOW_SIZE) )
clock = pygame.time.Clock()
projection_matrix = [[1,0,0],
                     [0,1,0],
                     [0,0,0]]

cube_points = [n for n in range(8)]
cube_points[0] = [[-1], [-1], [1]]
cube_points[1] = [[1],[-1],[1]]
cube_points[2] = [[1],[1],[1]]
cube_points[3] = [[-1],[1],[1]]
cube_points[4] = [[-1],[-1],[-1]]
cube_points[5] = [[1],[-1],[-1]]
cube_points[6] = [[1],[1],[-1]]
cube_points[7] = [[-1],[1],[-1]]


def multiply_m(a, b):
    a_rows = len(a)
    a_cols = len(a[0])

    b_rows = len(b)
    b_cols = len(b[0])
    # Dot product matrix dimentions = a_rows x b_cols
    product = [[0 for _ in range(b_cols)] for _ in range(a_rows)]

    if a_cols == b_rows:
        for i in range(a_rows):
            for j in range(b_cols):
                for k in range(b_rows):
                    product[i][j] += a[i][k] * b[k][j]
    else:
        print("INCOMPATIBLE MATRIX SIZES")
    return product


def connect_points(i, j, points):
    pygame.draw.line(window, (255, 255, 255), (points[i][0], points[i][1]) , (points[j][0], points[j][1]))

# Main Loop
scale = 100
angle_x = angle_y = angle_z = 0.28


cap = cv2.VideoCapture(0)
hand_detector = mp.solutions.hands.Hands()
drawing_utils = mp.solutions.drawing_utils
index_y = 0
while True:
    clock.tick(60)
    window.fill((0,0,0))
    rotation_x = [[1, 0, 0],
                    [0, cos(angle_x), -sin(angle_x)],
                    [0, sin(angle_x), cos(angle_x)]]

    rotation_y = [[cos(angle_y), 0, sin(angle_y)],
                    [0, 1, 0],
                    [-sin(angle_y), 0, cos(angle_y)]]

    rotation_z = [[cos(angle_z), -sin(angle_z), 0],
                    [sin(angle_z), cos(angle_z), 0],
                    [0, 0, 1]]

    points = [0 for _ in range(len(cube_points))]
    i = 0
    for point in cube_points:
        rotate_x = multiply_m(rotation_x, point)
        rotate_y = multiply_m(rotation_y, rotate_x)
        rotate_z = multiply_m(rotation_z, rotate_y)
        point_2d = multiply_m(projection_matrix, rotate_z)

        x = (point_2d[0][0] * scale) + WINDOW_SIZE/2
        y = (point_2d[1][0] * scale) + WINDOW_SIZE/2

        points[i] = (x,y)
        i += 1
        pygame.draw.circle(window, (255, 0, 0), (x, y), 5)

    connect_points(0, 1, points)
    connect_points(0, 3, points)
    connect_points(0, 4, points)
    connect_points(1, 2, points)
    connect_points(1, 5, points)
    connect_points(2, 6, points)
    connect_points(2, 3, points)
    connect_points(3, 7, points)
    connect_points(4, 5, points)
    connect_points(4, 7, points)
    connect_points(6, 5, points)
    connect_points(6, 7, points)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()




    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = hand_detector.process(rgb_frame)
    hands = output.multi_hand_landmarks
    if hands:
        for hand in hands:
            drawing_utils.draw_landmarks(frame, hand)
            landmarks = hand.landmark
            for id, landmark in enumerate(landmarks):
                x=(int)(landmark.x*frame_width)
                y=(int)(landmark.y*frame_height)
                if id==8:
                   #print(f" x= {x}  y={y}")
                   cv2.circle(img=frame, center=(x,y), radius=15, color=(0,255,0))
                   if (x>=14 and y>=24 and x<=213 and y<=239):
                       print("Top left")
                       angle_z -= ROTATE_SPEED
                   elif(x>213 and y>=24 and x<=411 and y<=239):
                       print("Top middle")
                       angle_x += ROTATE_SPEED
                   elif(x>411 and y>=24 and x<=610 and y<=239):
                       print("Top right")
                       angle_z += ROTATE_SPEED
                   elif(x>=14 and y>=239 and x<=213 and y<=453):
                       print("Bottom left")
                       angle_y += ROTATE_SPEED
                   elif(x>213 and y>=239 and x<=411 and y<=453):
                       print("Bottom middle")
                       angle_x -= ROTATE_SPEED
                   elif(x>411 and y>=239 and x<=610 and y<=453):
                       print("Bottom right")
                       angle_y -= ROTATE_SPEED
                   else:
                       print("Unhandled case")

    pygame.display.update()


    cv2.imshow('HandsNigga', frame)
    cv2.waitKey(1)
