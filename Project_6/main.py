import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation

camera = cv2.VideoCapture(0) # 1 for mac, 0 for windows

with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie_segmentation:
  bg_image = None
  while camera.isOpened():
    success, image = camera.read()
    if not success:
      print("Ignoring empty camera frame.")
      continue

    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    image.flags.writeable = False
    results = selfie_segmentation.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # 0.1 is the threshold value for if a pixel is in the background or foreground
    condition = np.stack(
      (results.segmentation_mask,) * 3, axis=-1) > 0.1

    if bg_image is None:
      bg_image = np.zeros(image.shape, dtype=np.uint8)
      bg_image[:] = (150, 150, 150)
    output_image = np.where(condition, image, bg_image)

    cv2.imshow('MediaPipe Segmentation', output_image)
    if cv2.waitKey(1) == ord('q'):
      break
camera.release()