import cv2
import mediapipe as mp
import numpy as np
import pickle
import pyttsx3

# Load trained model and LabelEncoder
with open("sign_model.pkl", "rb") as f:
    model, le = pickle.load(f)

# Text-to-Speech setup
engine = pyttsx3.init()
engine.setProperty('rate', 90)  # slower speech

# MediaPipe hands setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Webcam
cap = cv2.VideoCapture(0)

prev_prediction = ""
current_letter = ""
predicted_word = ""
sentence = ""
stable_count = 0
stable_threshold = 6

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    prediction = ""

    if result.multi_hand_landmarks:
        try:
            hand_landmarks = result.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            data = []
            for lm in hand_landmarks.landmark:
                data.extend([lm.x, lm.y, lm.z])

            data = np.array(data).reshape(-1, 3)
            if data.shape[0] != 21:
                raise ValueError("Not enough landmarks")

            data -= data[0]  # normalize by wrist
            data = data.flatten().reshape(1, -1)

            proba = model.predict_proba(data)[0]
            max_prob = np.max(proba)

            if max_prob > 0.65:
                prediction = le.inverse_transform([np.argmax(proba)])[0]

                if prediction == current_letter:
                    stable_count += 1
                else:
                    current_letter = prediction
                    stable_count = 0

                if stable_count >= stable_threshold and prediction != prev_prediction:
                    predicted_word += prediction
                    prev_prediction = prediction
                    stable_count = 0

        except Exception as e:
            print("⚠️ Error in prediction:", e)

    # Show prediction + word + sentence
    cv2.putText(frame, f"Current: {prediction}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(frame, f"Word: {predicted_word}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 2)

    cv2.putText(frame, f"Sentence: {sentence}", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("Sign Language to Text", frame)

    key = cv2.waitKey(1) & 0xFF

    # Press SPACE to confirm word and speak
    if key == ord(' '):
        if predicted_word and len(predicted_word) >= 2:
            engine.say(predicted_word)
            engine.runAndWait()
            sentence += predicted_word + " "
            predicted_word = ""
            prev_prediction = ""
            stable_count = 0

    # Press 'c' to clear sentence
    elif key == ord('c'):
        sentence = ""
        predicted_word = ""
        prev_prediction = ""
        current_letter = ""
        stable_count = 0

    # Press 'q' to quit
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
