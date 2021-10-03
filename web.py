from sklearn.preprocessing import MinMaxScaler
from spellchecker import SpellChecker
from demos import orchestrator, fig
from pathlib import Path
import streamlit as st
import mediapipe as mp
import joblib
import random
import cv2
import sys

spell = SpellChecker()
ml_model = joblib.load('rf_model.joblib') # Load image model

# Mediapipe hand utils
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

try:
    sys.path.remove(str(parent))
except ValueError:  # Already removed
    pass


VERSION = ".".join(st.__version__.split(".")[:2])

demo_pages = {
    "Method overview": orchestrator.show_examples,
    #"ASL table": fig.show_examples
}

avg = 0
count = 1

def draw_main_page():
    st.write(
        f"""
        # LiveSigns 👋
        """
    )
    current_cam = 1
    # st.write('Try switching to `cam 0` if you\'re on mac, and `cam 1` if windows.')
    cam_switch = st.checkbox('Switch camera input: (try using "Camera #1" on Mac, "Camera #2" on Windows)')

    if cam_switch:
        current_cam = 1-current_cam
        
    st.text('Current Camera: #'+str(current_cam + 1))

    st.write('Click `Run` to try out the live sign language translation. The ASL sign table can be found on the left; the `del` sign is used before entering a duplicate symbol; `space` is used to end a word.')
    st.write('(Hold up each symbol for at least one second for best results.)')
    run = st.checkbox('Run')
    landmark_on = st.checkbox('Overlay landmarks')
    
    FRAME_WINDOW = st.image([]) 
    cap = cv2.VideoCapture(current_cam)

    hand_word = ""
    prev_letter = ""
    freq_letter = 0
    displayed = False
    start_new_word = False
    auto_corrected = 0

    while run:
        with mp_hands.Hands(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    continue

                width, height = int(cap.get(3)), int(cap.get(4))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                #  image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

                image.flags.writeable = False
                font = cv2.FONT_HERSHEY_SIMPLEX # type: ignore
                results = hands.process(image)

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                if results.multi_hand_landmarks:
                    hand_landmarks, handedness = results.multi_hand_landmarks[0], results.multi_handedness[0]

                    if(landmark_on):
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
                        if freq_letter > 5 and not displayed:
                            if letter == 'space' and not start_new_word:
                                start_new_word = True
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
                    display_letter = letter if letter != 'del' else 'next'
                    image = cv2.putText(image, letter, # type: ignore
                                (textX, textY),
                                font, 3, (0, 0, 0), 5, cv2.LINE_AA # type: ignore
                    )

                text_size, _ = cv2.getTextSize(hand_word, font, 3, 2) # type: ignore
                image = cv2.rectangle( # type: ignore
                        image,
                        (int((width - text_size[0]) / 2), height - 100 - text_size[1] - 30),
                        (int((width + text_size[0]) / 2), height - 130 + text_size[1]),
                        (0, 0, 0), -1)

                display_word = hand_word

                if auto_corrected == 1:
                    cv2.putText( # type: ignore
                            image,
                            display_word,
                            (int((width - text_size[0]) / 2), height - 100),
                            font, 3, (0, 255, 255), 2, cv2.LINE_AA) # type: ignore
                elif auto_corrected == 2: # if correct!
                    cv2.putText( # type: ignore
                            image,
                            display_word,
                            (int((width - text_size[0]) / 2), height - 100),
                            font, 3, (0, 255, 0), 2, cv2.LINE_AA) # type: ignore
                else:
                    cv2.putText( # type: ignore
                            image,
                            display_word,
                            (int((width - text_size[0]) / 2), height - 100),
                            font, 3, (255, 255, 255), 2, cv2.LINE_AA) # type: ignore

                FRAME_WINDOW.image(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                #  if cv2.waitKey(5) & 0xFF == 27:
                    #  break
    
    else:
        cap.release()
        cv2.destroyAllWindows()
    


    #st.write(release_notes)


# Draw sidebar
pages = list(demo_pages.keys())

if len(pages):
    pages.insert(0, "LiveSigns")
    st.sidebar.title(f"ASL Recognition🎈")
    query_params = st.experimental_get_query_params()
    if "page" in query_params and query_params["page"][0] == "headliner":
        index = 1
    else:
        index = 0
    selected_demo = st.sidebar.radio("", pages, index, key="pages")
else:
    selected_demo = ""
st.sidebar.image("pic/asl.png", use_column_width=True)

# Draw main page
if selected_demo in demo_pages:
    demo_pages[selected_demo]()
else:
    draw_main_page()
    
