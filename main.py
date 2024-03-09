import cv2
import mediapipe as mp
import numpy as np
import time

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_face_detection = mp.solutions.face_detection

hands = mp_hands.Hands()
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

webcam = cv2.VideoCapture(0)

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

    # apply face detection model
    face_results = face_detection.process(img)

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
            if distance < 50:
                box_position = (thumb_tip + index_tip) / 2

    # Draw the box
    if box_position is not None:
        cv2.rectangle(img, (int(box_position[0] - 50), int(box_position[1] - 50)), 
                      (int(box_position[0] + 50), int(box_position[1] + 50)), (0, 255, 0), 2)

    # Draw the face detections
    if face_results.detections:
        for detection in face_results.detections:
            mp_drawing.draw_detection(img, detection)

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