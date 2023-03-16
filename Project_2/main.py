import cv2

# imports the cascades for detecting faces and eyes
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

camera = cv2.VideoCapture(0)

while True:
    _, frame = camera.read()

    # detects face patterns
    faces_rect = face_cascade.detectMultiScale(frame, scaleFactor=1.5, minNeighbors=5)

    for (x, y, w, h) in faces_rect:
        # draws a rectangle at each face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

        # takes the dimensions of the face and converts it into a region of interest
        roi = frame[y:y+h, x:x+w]

        # detects eye patterns
        eyes = eye_cascade.detectMultiScale(roi, scaleFactor=1.7, minNeighbors=8)
        for (ex,ey,ew,eh) in eyes:
            # draws a circle at each eye
            cv2.circle(frame,(ex + x + round(ew / 2),ey + y + round(eh / 2)),round(ew / 2),(0,255,0),-1)

    cv2.imshow('Camera', frame)
    
    if cv2.waitKey(1) == ord('q'):
        break