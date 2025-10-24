import cv2
import mediapipe as mp
import numpy as np
import os
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)
mp_drawing = mp.solutions.drawing_utils

labels = ['A', 'B', 'C', 'D', 'E','F','G','H','I','J','K','L','M','N','O','P',
          'Q','R','S','T','U','V','W','X','Y','Z','0','1','2','3','4','5','6','7','8','9']  # You can add more later

cap = cv2.VideoCapture(0)

while True:
    success, image = cap.read()
    if not success:
        continue

    image = cv2.flip(image, 1)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            data = []
            for lm in hand_landmarks.landmark:
                data.extend([lm.x, lm.y, lm.z])
            
            key = cv2.waitKey(1)
            if key != -1:
                key_char = chr(key).upper()
                if key_char in labels:
                    filename = f"data/{key_char}/{time.time()}.txt"
                    np.savetxt(filename, np.array(data))
                    print(f"Saved {key_char} data to {filename}")

    cv2.imshow("Collecting Sign Data", image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()