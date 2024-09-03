import cv2
import mediapipe as mp
import time
import random
import numpy as np
from PIL import Image
import HandTrackingModule as htm

pTime = 0
cTime = 0
cap = cv2.VideoCapture(0)
detector = htm.handDetector()

start_time = time.time()

countdown_finished = False

computer_choice = None

player_prediction = None

countdown_end_time = None

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img, start_time, countdown_finished)

    if not countdown_finished:
        with open('prediction.txt', 'r') as f:
            player_prediction = f.read().strip()

    if not countdown_finished:
        choices = ['Rock', 'Paper', 'Scissors']
        computer_choice = random.choice(choices)

    if computer_choice == 'Rock':
        computer_image = 'rock.png'
    elif computer_choice == 'Paper':
        computer_image = 'paper.png'
    elif computer_choice == 'Scissors':
        computer_image = 'scissors.png'

    computer_image = np.array(Image.open(computer_image).convert("RGBA"))
    computer_image = cv2.cvtColor(computer_image, cv2.COLOR_RGBA2BGRA)

    # Define the region where the overlay image will be placed
    x_offset, y_offset = 450, 150 
    overlay_width, overlay_height = 150, 150  

    computer_image = cv2.resize(computer_image, (overlay_width, overlay_height))

    overlay_color = computer_image[:, :, :3]
    overlay_alpha = computer_image[:, :, 3] / 255.0

    # Get the region of interest from the original image
    roi = img[y_offset:y_offset+overlay_height, x_offset:x_offset+overlay_width]

    for c in range(0, 3):
        roi[:, :, c] = (1.0 - overlay_alpha) * roi[:, :, c] + overlay_alpha * overlay_color[:, :, c]

    img[y_offset:y_offset+overlay_height, x_offset:x_offset+overlay_width] = roi


    cv2.putText(img, f'Player: {player_prediction}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    cv2.putText(img, f'Computer: {computer_choice}', (10, 120), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    current_time = time.time()
    elapsed_time = current_time - start_time
    remaining_time = 5 - int(elapsed_time)
    if remaining_time > 0:
        cv2.putText(img, str(remaining_time), (250, 300), cv2.FONT_HERSHEY_PLAIN, 15, (255, 0, 255), 3)
    else:
        if countdown_end_time is None:
            countdown_end_time = current_time
        if current_time - countdown_end_time < 1:
            cv2.putText(img, "Go!", (200, 300), cv2.FONT_HERSHEY_PLAIN, 10, (0, 255, 0), 3)
        countdown_finished = True

         # Outcome
        if countdown_finished:
            if player_prediction == computer_choice:
                result = "TIE"
            elif (player_prediction == "Rock" and computer_choice == "Scissors") or \
                (player_prediction == "Paper" and computer_choice == "Rock") or \
                (player_prediction == "Scissors" and computer_choice == "Paper"):
                result = "Player wins"
            else:
                result = "Computer wins"

            if result == "Player wins":
                cv2.putText(img, "Player", (250, 400), cv2.FONT_HERSHEY_PLAIN, 5, (0, 255, 0), 3)
                cv2.putText(img, "wins", (250, 440), cv2.FONT_HERSHEY_PLAIN, 5, (0, 255, 0), 3)
            elif result == "Computer wins":
                cv2.putText(img, "Computer", (250, 400), cv2.FONT_HERSHEY_PLAIN, 5, (0, 255, 0), 3)
                cv2.putText(img, "wins", (250, 440), cv2.FONT_HERSHEY_PLAIN, 5, (0, 255, 0), 3)
            else:
                cv2.putText(img, result, (250, 400), cv2.FONT_HERSHEY_PLAIN, 5, (0, 255, 0), 3)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, "FPS: " + str(int(fps)), (400, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)