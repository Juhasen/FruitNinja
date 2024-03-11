import cv2
import mediapipe as mp
import pyautogui

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands()

webcam = cv2.VideoCapture(0)

screen_width, screen_height = pyautogui.size()

while True:
    ret, img = webcam.read()
    img = cv2.flip(img, 1)

    # apply hands tracking model
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img)

    # draw annotations on the image
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(img, hand_landmarks, connections=mp_hands.HAND_CONNECTIONS)

            # get the position of the wrist
            wrist_x = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x
            wrist_y = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y

            # normalize to screen size
            screen_x = int(wrist_x * screen_width)
            screen_y = int(wrist_y * screen_height)

            # move the cursor
            pyautogui.dragTo(screen_x, screen_y, button="left")

    if ret:
        cv2.imshow("Juhas", img)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break

webcam.release()
cv2.destroyAllWindows()
