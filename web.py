import cv2
import streamlit as st
import mediapipe as mp
import joblib
from sklearn.preprocessing import MinMaxScaler

ml_model = joblib.load('models/rf_model.joblib')

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

avg = 0
count = 1


   
import sys
from pathlib import Path

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

try:
    sys.path.remove(str(parent))
except ValueError:  # Already removed
    pass

import streamlit as st

VERSION = ".".join(st.__version__.split(".")[:2])

from demos import orchestrator
from demos import fig

demo_pages = {
    "Pipeline": orchestrator.show_examples,
    "ASL table": fig.show_examples
}



contributors = []

# End release updates


def draw_main_page():
    st.write(
        f"""
        # Webcam Live Feed👋
        """
    )

    #st.write(intro)
    
    #st.title("Webcam Live Feed")
    run = st.checkbox('Run')
    FRAME_WINDOW = st.image([])
    cap = cv2.VideoCapture(0)

    while run:
    #_, frame = cap.read()
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #FRAME_WINDOW.image(frame)
    
    
        with mp_hands.Hands(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
            
                    continue


                image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

                image.flags.writeable = False
                results = hands.process(image)

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                          image,
                          hand_landmarks,
                          mp_hands.HAND_CONNECTIONS,
                          mp_drawing_styles.get_default_hand_landmarks_style(),
                          mp_drawing_styles.get_default_hand_connections_style())
                #cv2.imshow('MediaPipe Hands', image)
                #FRAME_WINDOW.image(image)
                FRAME_WINDOW.image(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                if cv2.waitKey(5) & 0xFF == 27:
                    break
    
    else:
        cap.release()
        cv2.destroyAllWindows
    


    #st.write(release_notes)


# Draw sidebar
pages = list(demo_pages.keys())

if len(pages):
    pages.insert(0, "Live result")
    st.sidebar.title(f"ASL Recognition🎈")
    query_params = st.experimental_get_query_params()
    if "page" in query_params and query_params["page"][0] == "headliner":
        index = 1
    else:
        index = 0
    selected_demo = st.sidebar.radio("", pages, index, key="pages")
else:
    selected_demo = ""

# Draw main page
if selected_demo in demo_pages:
    demo_pages[selected_demo]()
else:
    draw_main_page()
    
    
    
    
    
    
    
    
    













