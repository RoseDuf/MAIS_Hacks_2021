import cv2
import mediapipe as mp
import joblib
from sklearn.preprocessing import MinMaxScaler

ml_model = joblib.load('models/rf_model.joblib')

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

avg = 0
count = 1

# For webcam input:
cap = cv2.VideoCapture(1)
with mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:

        # mp_drawing.draw_landmarks(
        #     image,
        #     hand_landmarks,
        #     mp_hands.HAND_CONNECTIONS,
        #     mp_drawing_styles.get_default_hand_landmarks_style(),
        #     mp_drawing_styles.get_default_hand_connections_style())


        scaler = MinMaxScaler()

        originX = hand_landmarks.landmark[0].x
        originY = hand_landmarks.landmark[0].y
        originZ = hand_landmarks.landmark[0].z

        landmarkers = []
        for landmarker in hand_landmarks.landmark:
            landmarker.x -= originX
            landmarker.y -= originY
            landmarker.z -= originZ
            landmarkers.append([landmarker.x, landmarker.y, landmarker.z])
        
        scaled = scaler.fit_transform(landmarkers)

        output = []
        for coord in scaled:
            output.append(coord[0])
            output.append(coord[1])
            output.append(coord[2])
        
        letter = ml_model.predict([output])[0]
        print(letter)

    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
