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

from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

spell = SpellChecker()
ml_model = joblib.load('rf_model.joblib') # Load image model

st.set_page_config(layout='wide')

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

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.hand_word = ""
        self.prev_letter = ""
        self.freq_letter = 0
        self.displayed = False
        self.start_new_word = False
        self.auto_corrected = 0
        #  self.hands = mp_hands.Hands(
            #  min_detection_confidence=0.5,
            #  min_tracking_confidence=0.5)

    def transform(self, frame):
        with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
            image = frame.to_ndarray(format="bgr24")

            width, height, _ = image.shape
            #  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

            image.flags.writeable = False
            font = cv2.FONT_HERSHEY_SIMPLEX # type: ignore
            results = hands.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                hand_landmarks, handedness = results.multi_hand_landmarks[0], results.multi_handedness[0]

                #  mp_drawing.draw_landmarks(
                   #  image,
                   #  hand_landmarks,
                   #  mp_hands.HAND_CONNECTIONS,
                   #  mp_drawing_styles.get_default_hand_landmarks_style(),
                   #  mp_drawing_styles.get_default_hand_connections_style())


                # add text
                textX = int(hand_landmarks.landmark[0].x * width) + 150 # type: ignore
                textY = int(hand_landmarks.landmark[0].y * height) - 150 # type: ignore

                # calculate offset for right hand
                if handedness.classification[0].label == 'Right':
                    textX += 100

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
                    self.prev_letter = ''
                elif letter == self.prev_letter:
                    self.freq_letter += 1
                    if self.freq_letter > 10 and not self.displayed:
                        if letter == 'space' and not self.start_new_word:
                            self.start_new_word = True
                            misspelled = list(spell.unknown([self.hand_word]))

                            if len(misspelled) != 0:
                                self.hand_word = spell.correction(misspelled[0])
                                self.auto_corrected = 1
                            else:
                                self.auto_corrected = 2

                        elif self.start_new_word:
                            self.start_new_word = False
                            self.auto_corrected = 0
                            self.hand_word = letter
                        else:
                            self.hand_word += letter

                        self.displayed = not self.displayed
                elif letter != self.prev_letter:
                    #  hand_word += letter
                    self.freq_letter = 0
                    self.displayed = False

                self.prev_letter = letter

                #  if letter == 'space':
                    #  image = cv2.putText(image, '', # type: ignore
                                #  (textX, textY),
                                #  font, 3, (0, 0, 0), 5, cv2.LINE_AA # type: ignore
                    #  )
                #  else:
                display_letter = letter if letter != 'del' else 'next'
                image = cv2.putText(image, letter, # type: ignore
                            (textX, textY),
                            font, 2, (0, 0, 0), 5, cv2.LINE_AA # type: ignore
                )

            text_size, _ = cv2.getTextSize(self.hand_word, font, 2, 2) # type: ignore
            image = cv2.rectangle( # type: ignore
                    image,
                    (int((height - text_size[0]) / 2), width - 60 - text_size[1] - 10),
                    (int((height + text_size[0]) / 2), width - 60 + 10),
                    (0, 0, 0), -1)

            display_word = self.hand_word

            if self.auto_corrected == 1:
                cv2.putText( # type: ignore
                        image,
                        display_word,
                        (int((height - text_size[0]) / 2), width - 60),
                        font, 2, (0, 255, 255), 2, cv2.LINE_AA) # type: ignore
            elif self.auto_corrected == 2: # if correct!
                cv2.putText( # type: ignore
                        image,
                        display_word,
                        (int((height - text_size[0]) / 2), width - 60),
                        font, 2, (0, 255, 0), 2, cv2.LINE_AA) # type: ignore
            else:
                cv2.putText( # type: ignore
                        image,
                        display_word,
                        (int((height - text_size[0]) / 2), width - 60),
                        font, 2, (255, 255, 255), 2, cv2.LINE_AA) # type: ignore


            return image

def draw_main_page():
    st.write(
        f"""
        # LiveSigns 👋
        """
    )

    st.write('Click `Run` to try out the live sign language translation. The ASL sign table can be found on the left; the `del` sign is used before entering a duplicate symbol; `space` is used to end a word.')
    st.write('(Hold up each symbol for at least one second for best results.)')
    run = st.checkbox('Run')

    #  FRAME_WINDOW = st.image([]) 
    #  cap = cv2.VideoCapture(current_cam)

    if run:
        webrtc_streamer(key="special_key_or_something", video_transformer_factory=VideoTransformer)

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
    
