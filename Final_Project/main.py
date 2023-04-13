import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# This uses video capture 1, Windows usually uses 0
cap = cv2.VideoCapture(0)

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

    keypoints = []
    if results.multi_hand_landmarks:
      
      for hand_landmarks in results.multi_hand_landmarks:
        for data_point in hand_landmarks.landmark:
          keypoints.append({'x': data_point.x, 'y': data_point.y})

      #  mp_drawing.draw_landmarks(
       #     image,
        #    hand_landmarks,
        #    mp_hands.HAND_CONNECTIONS,
        #    mp_drawing_styles.get_default_hand_landmarks_style(),
         #   mp_drawing_styles.get_default_hand_connections_style())



        rect_x = int(keypoints[10]['x'] * image.shape[1])
        rect_y = int(keypoints[10]['y'] * image.shape[0])

        cv2.rectangle(image, (rect_x, rect_y), (rect_x + 50, rect_y + 50), (255, 0, 0), -1)
        # sets keypoints to be an empty array so it can have the second hand's data if necessary
        keypoints = []
    
      

    cv2.imshow('Tic-Tac-Toe', cv2.flip(image, 1))
    if cv2.waitKey(1) == ord('q'):
        break
    
cv2.destroyAllWindows()