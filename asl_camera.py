import pyvirtualcam
import cv2
import mediapipe as mp
import time

import mediapipe as mp
import joblib
from sklearn.preprocessing import MinMaxScaler

from spellchecker import SpellChecker

spell = SpellChecker()

ml_model = joblib.load('models/rf_model.joblib')


mp_drawing = mp.solutions.drawing_utils # type: ignore
mp_drawing_styles = mp.solutions.drawing_styles # type: ignore
mp_hands = mp.solutions.hands # type: ignore

LETTERS = ['a', 'b', 'c', 'd', 'e', 'f', 'g']

FREQ_DELAY = 10

in_debug = input('Debug mode? (y/n): ')

debug = False
if in_debug == 'y':
    debug = True

cap = cv2.VideoCapture(1) # type: ignore
fmt = pyvirtualcam.PixelFormat.BGR

hand_word = ""
prev_letter = ""
freq_letter = 0
displayed = False
start_new_word = False
auto_corrected = 0

with pyvirtualcam.Camera(width=1280, height=720, fps=20, fmt=fmt) as cam:
    with mp_hands.Hands(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
        while True:
            ret_val, image = cap.read()

            new_frame_time = time.time()
 
            image = cv2.resize(image, (1280, 720), interpolation=cv2.BORDER_DEFAULT) # type: ignore
            # Flip the image horizontally for a later selfie-view display, and convert
            # the BGR image to RGB.
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB) # type: ignore
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            results = hands.process(image)

            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # type: ignore

            font = cv2.FONT_HERSHEY_SIMPLEX # type: ignore
            width, height = int(cap.get(3)), int(cap.get(4))

            if results.multi_hand_landmarks:
              for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # draw landmarks
                if debug:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

                # add text
                textX = int(hand_landmarks.landmark[0].x * width) - 150 # type: ignore
                textY = int(hand_landmarks.landmark[0].y * height) - 100 # type: ignore

                # calculate offset for right hand
                if handedness.classification[0].label == 'Right':
                    textX += 250

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

                if letter == 'del':
                    prev_letter = ''
                elif letter == prev_letter:
                    freq_letter += 1
                    if freq_letter > FREQ_DELAY and not displayed:
                        if letter == 'space' and not start_new_word:
                            start_new_word = True

                            if hand_word == 'jinho' or hand_word == 'eduarard' or hand_word == 'rose' or hand_word == 'bridget':
                                hand_word = hand_word[0].upper() + hand_word[1:]


                            misspelled = list(spell.unknown([hand_word]))

                            if len(misspelled) != 0:
                                hand_word = spell.correction(misspelled[0])
                                auto_corrected = 1
                            else:
                                auto_corrected = 2

                        elif start_new_word:
                            start_new_word = False
                            auto_corrected = 0
                            hand_word = letter
                        else:
                            hand_word += letter

                        displayed = not displayed
                elif letter != prev_letter:
                    #  hand_word += letter
                    freq_letter = 0
                    displayed = False

                prev_letter = letter

                    #  if letter == 'space':
                        #  image = cv2.putText(image, '', # type: ignore
                                    #  (textX, textY),
                                    #  font, 3, (0, 0, 0), 5, cv2.LINE_AA # type: ignore
                        #  )
                    #  else:
                image = cv2.putText(image, letter, # type: ignore
                            (textX, textY),
                            font, 3, (0, 0, 0), 5, cv2.LINE_AA # type: ignore
                )

                #
            text_size, _ = cv2.getTextSize(hand_word, font, 3, 2) # type: ignore
            image = cv2.rectangle( # type: ignore
                    image,
                    (int((width - text_size[0]) / 2), height - 100 - text_size[1] - 30),
                    (int((width + text_size[0]) / 2), height - 130 + text_size[1]),
                    (0, 0, 0), -1)

            if auto_corrected == 1:
                cv2.putText( # type: ignore
                        image,
                        hand_word,
                        (int((width - text_size[0]) / 2), height - 100),
                        font, 3, (0, 255, 255), 2, cv2.LINE_AA) # type: ignore
            elif auto_corrected == 2: # if correct!
                cv2.putText( # type: ignore
                        image,
                        hand_word,
                        (int((width - text_size[0]) / 2), height - 100),
                        font, 3, (0, 255, 0), 2, cv2.LINE_AA) # type: ignore
            else:
                cv2.putText( # type: ignore
                        image,
                        hand_word,
                        (int((width - text_size[0]) / 2), height - 100),
                        font, 3, (255, 255, 255), 2, cv2.LINE_AA) # type: ignore




                #  cv2.putText(image, LETTERS[random.randint(0, 5)], # type: ignore
                            #  (textX, textY),
                            #  font, 3, (0, 0, 0), 2, cv2.LINE_AA # type: ignore
                #  )

            #  completed_word = 'word here'
            #  text_size, _ = cv2.getTextSize(completed_word, font, 3, 2) # type: ignore
            #  cv2.rectangle( # type: ignore
                    #  image,
                    #  (int((width - text_size[0]) / 2), height - 100 - text_size[1] - 30),
                    #  (int((width + text_size[0]) / 2), height - 130 + text_size[1]),
                    #  (0, 0, 0), -1)
            #  #  cv2.putText( # type: ignore
                    #  #  image,
                    #  #  'word here',
                    #  #  (int((width - text_size[0]) / 2), height - 100),
                    #  #  font, 3, (255, 255, 255), 2, cv2.LINE_AA) # type: ignore
            cv2.imshow('Preview', image) # type: ignore

            cam.send(image)
            cam.sleep_until_next_frame()

            cv2.destroyAllWindows() # type: ignore
