import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
from sklearn.neighbors import KNeighborsClassifier

mp_holistic =mp.solutions.holistic
mp_drawing =mp.solutions.drawing_utils

knn = None
X_train = []  
y_train = []
def points_detection(image,model):
  image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
  image.flags.writeable=False
  results=model.process(image)
  image.flags.writeable=True
  image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
  return image, results

def extract_features(landmarks):
    if landmarks is None:
        return [0.0] * num_features 

    features = []

    for landmark in landmarks.landmark:
        features.append(landmark.x)
        features.append(landmark.y)
        features.append(landmark.z)
    return np.array(features)

def calculate_angle(point1, point2, point3):
    vector1 = np.array(point1) - np.array(point2)
    vector2 = np.array(point3) - np.array(point2)
    dot_product = np.dot(vector1, vector2)
    norm_product = np.linalg.norm(vector1) * np.linalg.norm(vector2)
    angle = np.arccos(dot_product / norm_product)
    return angle

def train_classifier(X, y):
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X, y)
    return knn

num_features = 0 

def draw_landmarks(image,results):
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks,mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks,mp_holistic.HAND_CONNECTIONS)
    for hand_landmarks in [results.left_hand_landmarks, results.right_hand_landmarks]:
        if hand_landmarks:
            landmarks = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark], np.float32)
            landmarks *= np.array([image.shape[1], image.shape[0]])
            landmarks = landmarks.astype(np.int32)
            x, y, w, h = cv2.boundingRect(landmarks)
            cv2.rectangle(image, (x-20, y-20), (x + w+20, y + h+20), (0, 255, 0), 2)

data_dir = 'dataset'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
gestures = ['Hello', 'Love', 'I love you']
fps = 30
gesture_counters = {gesture: 0 for gesture in gestures}

def collect_gesture_data(gesture):
    global gesture_counters
    cap = cv2.VideoCapture(0)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            image, results = points_detection(frame, holistic)

            countdown = fps - gesture_counters[gesture]
            cv2.putText(image, f"Collecting data for {gesture} gesture - Frame {countdown} / {fps}",
                        (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


            draw_landmarks(image, results)
            cv2.imshow("OpenCV Feed", image)

            if gesture_counters[gesture] < fps:
                folder_path = os.path.join(data_dir, gesture)
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                image_filename = os.path.join(folder_path, f'{gesture}_{gesture_counters[gesture]}.jpg')
                cv2.imwrite(image_filename, image)
                gesture_counters[gesture] += 1

                time.sleep(0.01)
                

            if cv2.waitKey(10) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()

for gesture in gestures:
    print(f"Collecting data for {gesture} gesture. Please make the gesture and press 'q' when done.")
    collect_gesture_data(gesture)
if X_train and y_train:
    knn = train_classifier(X_train, y_train)
    print("KNN classifier is trained.")

# setting up camera
cap=cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5,min_tracking_confidence=0.5)as holistic:
    while cap.isOpened():
        ret,frame=cap.read()

        image,results=points_detection(frame,holistic)
        
        draw_landmarks(image,results)
        features = extract_features(results.left_hand_landmarks)
        if knn is not None:
            predicted_gesture = knn.predict([features])
            print("Predicted Gesture:", predicted_gesture[0])
        else:
            cv2.putText(image, "KNN classifier is not trained.", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imshow("open cv feed",image)

        if cv2.waitKey(10)& 0xFF==ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()