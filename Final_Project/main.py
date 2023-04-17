import cv2
import mediapipe as mp
import math

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# This uses video capture 1, Windows usually uses 0
cap = cv2.VideoCapture(1)

tileSize = 150
selectorSize = 20

players = [{'x': 0, 'y': 0, 'symbol': 'X', 'old_index_distance': 0, 'index_distance': 0, 'is_clicking': False},
           {'x': 0, 'y': 0, 'symbol': 'O', 'old_index_distance': 0, 'index_distance': 0, 'is_clicking': False}]

grid = [
  ['', '', ''],
  ['', '', ''],
  ['', '', ''],
]

def detectMousePosition(event, x, y, flags, params):
   if event == cv2.EVENT_LBUTTONDBLCLK:
      global mouseX
      global mouseY
      mouseX, mouseY = x, y

def drawGrid(image):
  for y in range(0, 3):
      for x in range(0, 3):
        color = (100, 100, 100)

        for player in players:
          playerX = player['x']
          playerY = player['y']

          squareX = int(image.shape[1] / 3) + (tileSize * x)
          squareY = int(image.shape[0] / 5) + (tileSize * y)

          if playerX <= squareX + tileSize and\
            playerX + selectorSize >= squareX and\
            playerY <= squareY + tileSize and\
            playerY + selectorSize >= squareY:
            if player['is_clicking']:
              color = (0, 0, 255)
              player['is_clicking'] = False
            break
          
        cv2.rectangle(image, (squareX, squareY), (squareX + tileSize, squareY + tileSize), color, -1)
        cv2.rectangle(image, (squareX, squareY), (squareX + tileSize, squareY + tileSize), (0, 0, 0), 2)
          

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      continue

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    cv2.rectangle(image, (0, 0), (image.shape[1], image.shape[0]), (100, 100, 100), -1)

    index = 0

    drawGrid(image)

    # check distance between points 9 and 6 (index finger) (if has decreased)

    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        keypoints = []

        shape_color = (255, 0, 0)

        if index >= 2:
          index = 1

        for data_point in hand_landmarks.landmark:
          keypoints.append({'x': data_point.x, 'y': data_point.y})

        players[index]['x'] = int(keypoints[10]['x'] * image.shape[1])
        players[index]['y'] = int(keypoints[10]['y'] * image.shape[0])
        
        players[index]['old_index_distance'] = players[index]['index_distance']
        players[index]['index_distance'] = math.sqrt((keypoints[8]['x'] - keypoints[5]['x'])**2 + (keypoints[8]['y'] - keypoints[5]['y'])**2)

        if players[index]['index_distance'] + 0.02 < players[index]['old_index_distance']:
          players[index]['is_clicking'] = True

        cv2.rectangle(image, (players[index]['x'], players[index]['y']), (players[index]['x'] + selectorSize, players[index]['y'] + selectorSize), shape_color, -1)
        cv2.rectangle(image, (players[index]['x'], players[index]['y']), (players[index]['x'] + selectorSize, players[index]['y'] + selectorSize), (0, 0, 0), 2)
        
      
        index += 1

    cv2.imshow('Tic-Tac-Toe', cv2.flip(image, 1))
    cv2.setMouseCallback('Tic-Tac-Toe',detectMousePosition)
    if cv2.waitKey(1) == ord('q'):
        break
    
cv2.destroyAllWindows()