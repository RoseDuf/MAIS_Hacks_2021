import cv2
import mediapipe as mp
import pandas as pd
import os

# Create the training database

# Loop through all the files in a folder 
trainDirectory = 'asl_alphabet_train/asl_alphabet_train'
names = []
folders = []
for root, subdirectories, files in os.walk(trainDirectory):
    for subdirectory in subdirectories:
        # loop thru all the files in a folder 
        filenames = []
        directory = os.path.join(root, subdirectory)
        for filename in os.listdir(directory):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                filenames.append(os.path.join(directory, filename))
            else:
                continue
        names.append(filenames)
        folders.append(subdirectory.lower())

# Create dataframe structure
dfColumns = ['LetterLabel']
for i in range(21):
    dfColumns.append(f'X{i}')
    dfColumns.append(f'Y{i}')
    dfColumns.append(f'Z{i}')

dfObj = pd.DataFrame(columns=dfColumns)

mp_hands = mp.solutions.hands

MAX_FILES = 200

for i in range(29):
    trainFiles = names[i]
    file_count = 0
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5) as hands:
        for idx, file in enumerate(trainFiles):
            print(f'{folders[i]}: {file_count / MAX_FILES * 100}%')
            image = cv2.imread(file)
            # Convert the BGR image to RGB before processing.
            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if not results.multi_hand_landmarks:
                continue 
            file_count += 1
            if file_count <= MAX_FILES:       
                landmarklist = {'LetterLabel': folders[i]}
                for id, landmark in enumerate(results.multi_hand_landmarks[0].landmark):
                    landmarklist[f'X{id}'] = landmark.x
                    landmarklist[f'Y{id}'] = landmark.y
                    landmarklist[f'Z{id}'] = landmark.z
                
                dfObj = dfObj.append([landmarklist], ignore_index=True)
            else:
                break
        
dfObj.to_csv('training.csv')