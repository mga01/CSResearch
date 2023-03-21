

# WORK IN PROGRESS

import face_recognition
import cv2
import numpy

image = face_recognition.load_image_file("./img/Joe_Biden.jpg")
face_landmarks_list = face_recognition.face_locations(image, model="cnn")

picture_of_joe_biden = face_recognition.load_image_file("Joe_Biden.jpg")
joe_biden_encoding = face_recognition.face_encodings(picture_of_joe_biden)[0]


# Create arrays of known face encodings and their names
known_face_encodings = [
    joe_biden_encoding
]
known_face_names = [
    "Joe Biden"
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True


for (top, right, bottom, left) in face_landmarks_list:

        cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)
        #font = cv2.FONT_HERSHEY_DUPLEX
        # cv2.putText(image, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

face_locations = face_recognition.face_locations(image)
face_encodings = face_recognition.face_encodings(image, face_locations)

face_names = []
for face_encoding in face_encodings:
    # See if the face is a match for the known face(s)
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
    name = "Unknown"

    # # If a match was found in known_face_encodings, just use the first one.
    # if True in matches:
    #     first_match_index = matches.index(True)
    #     name = known_face_names[first_match_index]

    # Or instead, use the known face with the smallest distance to the new face
    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
    best_match_index = numpy.argmin(face_distances)
    if matches[best_match_index]:
        name = known_face_names[best_match_index]

    face_names.append(name)


while True:
    cv2.imshow('Image', image)
    
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()