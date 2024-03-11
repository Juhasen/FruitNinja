import cv2
import mediapipe as mp
import numpy as np
import time

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands()

webcam = cv2.VideoCapture(0)

# Load the image
img_to_move = cv2.imread('/home/krystian/Repos/FruitNinja/Sigmastycznie.jpg')  
if img_to_move is None:
    print("Could not load image")
    exit(1)

# Initialize the box's position
box_position = None

# Initialize the FPS counter
frame_counter = 0
start_time = time.time()

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

            # Calculate the distance between the thumb tip (index 4) and the index finger tip (index 8)
            thumb_tip = np.array([hand_landmarks.landmark[4].x, hand_landmarks.landmark[4].y]) * np.array([img.shape[1], img.shape[0]])
            index_tip = np.array([hand_landmarks.landmark[8].x, hand_landmarks.landmark[8].y]) * np.array([img.shape[1], img.shape[0]])
            distance = np.linalg.norm(thumb_tip - index_tip)

            # If the distance is below a certain threshold, update the box's position to the midpoint of the thumb tip and the index finger tip
            if distance < 25:
                box_position = (thumb_tip + index_tip) / 2

    # Overlay the image on the webcam feed at the box_position
    if box_position is not None:
        img_to_move_resized = cv2.resize(img_to_move, (int(distance), int(distance)))
        x1 = int(box_position[0] - distance / 2)
        x2 = x1 + img_to_move_resized.shape[1]
        y1 = int(box_position[1] - distance / 2)
        y2 = y1 + img_to_move_resized.shape[0]

        # Adjust the positions and the slice of img_to_move_resized if they are outside the image boundaries
        if x1 < 0:
            img_to_move_resized = img_to_move_resized[:, -x1:]
            x1 = 0
        if y1 < 0:
            img_to_move_resized = img_to_move_resized[-y1:, :]
            y1 = 0
        if x2 > img.shape[1]:
            img_to_move_resized = img_to_move_resized[:, :(img.shape[1]-x1)]
            x2 = img.shape[1]
        if y2 > img.shape[0]:
            img_to_move_resized = img_to_move_resized[:(img.shape[0]-y1), :]
            y2 = img.shape[0]

        img[y1:y2, x1:x2] = img_to_move_resized

    # Calculate and display the FPS
    frame_counter += 1
    elapsed_time = time.time() - start_time
    fps = frame_counter / elapsed_time
    cv2.putText(img, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if ret:
        cv2.imshow("Juhas", img)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break

webcam.release()
cv2.destroyAllWindows()