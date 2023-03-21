import face_recognition
import cv2

image = face_recognition.load_image_file("./img/Joe_Biden.jpg")
face_landmarks_list = face_recognition.face_locations(image, model="cnn")

for (top, right, bottom, left) in face_landmarks_list:

        cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)
        #font = cv2.FONT_HERSHEY_DUPLEX
        # cv2.putText(image, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

while True:
    cv2.imshow('Image', image)
    
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()