import cv2
import numpy as np
import mediapipe as mp
import math

from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

vol_range = volume.GetVolumeRange()
print(vol_range)
min_vol, max_vol, _ = vol_range

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    x, y, c = frame.shape
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(framergb)
    if result.multi_hand_landmarks:
        
        for handslms in result.multi_hand_landmarks:
            thumb = handslms.landmark[4]
            thumb_coords = [int(thumb.x * y), int(thumb.y * x)]
            index_finger = handslms.landmark[8]
            index_coords = [int(index_finger.x * y), int(index_finger.y * x)]

            cv2.line(frame, thumb_coords, index_coords, (255, 0, 0), 2)

            eu_dist=math.sqrt((thumb_coords[1]-index_coords[1])**2+(thumb_coords[0]-index_coords[0])**2) 
            percent=max(min(eu_dist / 200, 1), 0) 
            
            vol = (percent / 1) * (max_vol - min_vol) + min_vol
            volume.SetMasterVolumeLevel(vol, None)

            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)
        
    cv2.imshow("Output", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()