import cv2
import mediapipe as mp
import math
import tkinter

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

root = tkinter.Tk()
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
root.destroy()

# initialize video capture object to read video from external webcam
cap = cv2.VideoCapture(1)
# if there is no external camera then take the built-in camera
if not cap.read()[0]:
    cap = cv2.VideoCapture(0)

tileSize = int(screen_width / 8)
selectorSize = 20

players = [{'x': 0, 'y': 0, 'symbol': 'X', 'old_index_distance': 0, 'index_distance': 0, 'is_clicking': False, 'color': (0, 255, 0)},
           {'x': 0, 'y': 0, 'symbol': 'O', 'old_index_distance': 0, 'index_distance': 0, 'is_clicking': False, 'color': (0, 0, 255)}]

grid = [
  ['', '', ''],
  ['', '', ''],
  ['', '', ''],
]

def rectangle(x, y, w, h, color, fill):
  cv2.rectangle(image, (x, y), (x + w, y + h), (color), fill)

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
              color = (75, 75, 75)
              grid[y][x] = player['symbol']
              player['is_clicking'] = False
            break
          
        cv2.rectangle(image, (squareX, squareY), (squareX + tileSize, squareY + tileSize), color, -1)
        cv2.rectangle(image, (squareX, squareY), (squareX + tileSize, squareY + tileSize), (0, 0, 0), 2)

        cv2.putText(image, grid[y][x], (squareX + 50, squareY + int(tileSize / 1.5)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2, cv2.LINE_AA)

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.5,
    ) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      continue

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    cv2.rectangle(image, (0, 0), (screen_width, screen_height), (100, 100, 100), -1)

    index = 0

    drawGrid(image)
    
    # removes the outer lines of boxes on the edges of the board
    rectangle(int(image.shape[1] / 3) + (tileSize * 3) - 2, int(image.shape[0] / 5) - 1, 3, tileSize * 3 + 2, (100, 100, 100), -1)
    rectangle(int(image.shape[1] / 3) - 2, int(image.shape[0] / 5) - 1, 3, tileSize * 3 + 2, (100, 100, 100), -1)
    rectangle(int(image.shape[1] / 3) - 2, int(image.shape[0] / 5) - 1, tileSize * 3 + 2, 3, (100, 100, 100), -1)
    rectangle(int(image.shape[1] / 3) - 2, int(image.shape[0] / 5) + (tileSize * 3) - 1, tileSize * 3 + 2, 3, (100, 100, 100), -1)

    # check distance between points 9 and 6 (index finger) (if has decreased)
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        keypoints = []

        shape_color = players[index]['color']

        if index >= 2:
          break

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
    
    image = cv2.resize(image, (screen_width, screen_height))
    cv2.imshow('Tic-Tac-Toe', cv2.flip(image, 1))
    cv2.setMouseCallback('Tic-Tac-Toe', detectMousePosition)  

    if cv2.waitKey(1) == ord('q'):
        break
    
cv2.destroyAllWindows()