# --- imports ---

import math
import cv2
import cvzone
from cvzone.ColorModule import ColorFinder
import numpy as np
 
# --- main ---

# Initialize video capture
video = cv2.VideoCapture('video.mp4')
 


# Create the color finder object
my_color_finder = ColorFinder(False)

target_object_upper_lower_bounds = {'hmin': 0, 'smin': 200, 'vmin': 224, 'hmax': 30, 'smax': 226, 'vmax': 255}


 
# Lists of x and y position coordinates
posListX, posListY = [], []
# A list of all possible x coordinate values, used to calculate y
xList = [item for item in range(0, 1300)]

 
# create basketball hoop boundaries
hoop_boundaries = {'x1': 600, 'y1': 440, 'x2': 750, 'y2': 450}

while True:
    # Extract frame
    _, frame = video.read()
    # frame = cv2.imread("Ball.png")
    frame = frame[0:900, :]
    # cv2.rectangle(frame, (hoop_boundaries['x1'], hoop_boundaries['y1']), (hoop_boundaries['x2'],hoop_boundaries['y2']), (255,0, 0), -1)
 
    # Find the target object's color
    target_object_color, mask = my_color_finder.update(frame, target_object_upper_lower_bounds)
    # Find location of the target object
    target_contours, contours = cvzone.findContours(frame, mask, minArea=500)
 
    prediction = False  

    # Find center points of target
    if contours:
        posListX.append(contours[0]['center'][0])
        posListY.append(contours[0]['center'][1])
 
    if posListX:
        # Polynomial regression equation: y = Ax^2 + Bx + C
        # Find the coefficients
        A, B, C = np.polyfit(posListX, posListY, 2)

        # Drawing a line between each point
        for i, (p_x, p_y) in enumerate(zip(posListX, posListY)):
            current_position = (p_x, p_y)
            cv2.circle(target_contours, current_position, 10, (0, 255, 0), -1)
            if i == 0:
                cv2.line(target_contours, current_position, current_position, (255, 0, 0), 5)
            else:
                # (posListX[i - 1], posListY[i - 1]) is the previous position
                cv2.line(target_contours, current_position, (posListX[i - 1], posListY[i - 1]), (255, 0, 0), 5)
 
        # Predicting future positions of the target object
        for x in xList:
            y = int(A * x ** 2 + B * x + C)
            cv2.circle(target_contours, (x, y), 2, (0, 0, 255), -1)

            # Checks to see if future position is inside of the hoop

            if(x > hoop_boundaries['x1'] and x < hoop_boundaries['x1'] + (hoop_boundaries['x2'] - hoop_boundaries['x1']) and
               y > hoop_boundaries['y1'] and y < hoop_boundaries['y1'] + (hoop_boundaries['y2'] - hoop_boundaries['y1'])):
                prediction = True
    

        if prediction:
             cvzone.putTextRect(target_contours, "Basket", (50, 150), scale=5, thickness=5, colorR=(0, 200, 0), offset=20)
        else:
             cvzone.putTextRect(target_contours, "No Basket", (50, 150), scale=5, thickness=5, colorR=(0, 0, 200), offset=20)
 
    # Display
    target_contours = cv2.resize(target_contours, (0, 0), None, 0.7, 0.7)
    # cv2.imshow("Image", img)
    cv2.imshow("Video", target_contours)
    cv2.waitKey(100)
