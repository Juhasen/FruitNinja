import cv2
import mediapipe as mp
import numpy as np
import time

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands()

webcam = cv2.VideoCapture(0)

while True:
    ret, img = webcam.read()
    img = cv2.flip(img, 1)

    # apply hands tracking model
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img)

    hand_counter = 0
    
    num_of_fingers = 0

    # draw annotations on the image
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get the classification results for the hand
            hand_classification = results.multi_handedness[hand_counter]
            
            # Determine if the hand is a left or right hand
            hand_type = hand_classification.classification[0].label  # 'Left' or 'Right'
            
            # logic to see how many fingers are up (default: right hand)
            fingers = [0, 0, 0, 0, 0]
            if hand_type == "Left":
                if hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x:
                    fingers[0] = 1
            elif hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
                fingers[0] = 1
            if hand_landmarks.landmark[8].y < hand_landmarks.landmark[7].y:
                fingers[1] = 1
            if hand_landmarks.landmark[12].y < hand_landmarks.landmark[11].y:
                fingers[2] = 1
            if hand_landmarks.landmark[16].y < hand_landmarks.landmark[15].y:
                fingers[3] = 1
            if hand_landmarks.landmark[20].y < hand_landmarks.landmark[19].y:
                fingers[4] = 1

            num_of_fingers += sum(fingers)

            hand_counter += 1
            
            # draw the hand landmarks
            mp_drawing.draw_landmarks(img, hand_landmarks, connections=mp_hands.HAND_CONNECTIONS)
    
    # print in the right bottom corner the number of fingers
    cv2.putText(img, f"Fingers: {num_of_fingers}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    if ret:
        cv2.imshow("Juhas", img)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break

webcam.release()
cv2.destroyAllWindows()
