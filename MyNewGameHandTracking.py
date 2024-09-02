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

# Initialize start_time once
start_time = time.time()

# Flag to indicate if countdown has finished
countdown_finished = False

# Variable to store the computer's choice
computer_choice = None

# Variable to store the player's prediction
player_prediction = None

# Variable to store the time when the countdown finished
countdown_end_time = None

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img, start_time, countdown_finished)

    # Read the prediction from the file if the countdown has not finished
    if not countdown_finished:
        with open('prediction.txt', 'r') as f:
            player_prediction = f.read().strip()

    # Generate a random choice for the computer if the countdown has not finished
    if not countdown_finished:
        choices = ['Rock', 'Paper', 'Scissors']
        computer_choice = random.choice(choices)

    # Map the computer choice to the corresponding image file
    if computer_choice == 'Rock':
        computer_image = 'rock.png'
    elif computer_choice == 'Paper':
        computer_image = 'paper.png'
    elif computer_choice == 'Scissors':
        computer_image = 'scissors.png'

    # Load the computer's choice image
    computer_image = np.array(Image.open(computer_image).convert("RGBA"))
    computer_image = cv2.cvtColor(computer_image, cv2.COLOR_RGBA2BGRA)

    # Define the region where the overlay image will be placed
    x_offset, y_offset = 450, 150  # Top-left corner of the region
    overlay_width, overlay_height = 150, 150  # Size of the overlay image

    # Resize the overlay image to fit the defined region
    computer_image = cv2.resize(computer_image, (overlay_width, overlay_height))

    # Split the overlay image into its color and alpha channels
    overlay_color = computer_image[:, :, :3]
    overlay_alpha = computer_image[:, :, 3] / 255.0

    # Get the region of interest from the original image
    roi = img[y_offset:y_offset+overlay_height, x_offset:x_offset+overlay_width]

    # Blend the overlay image with the region of interest
    for c in range(0, 3):
        roi[:, :, c] = (1.0 - overlay_alpha) * roi[:, :, c] + overlay_alpha * overlay_color[:, :, c]

    # Put the blended region back into the original image
    img[y_offset:y_offset+overlay_height, x_offset:x_offset+overlay_width] = roi

    # Display the player's choice
    cv2.putText(img, f'Player: {player_prediction}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    # Display the computer's choice
    cv2.putText(img, f'Computer: {computer_choice}', (10, 120), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    # Get the current time
    current_time = time.time()
    # Calculate the elapsed time since the start of the program
    elapsed_time = current_time - start_time
    # Calculate the remaining time
    remaining_time = 5 - int(elapsed_time)
    # Check if the remaining time is greater than 0
    if remaining_time > 0:
        # Display the remaining time on the image
        cv2.putText(img, str(remaining_time), (250, 300), cv2.FONT_HERSHEY_PLAIN, 15, (255, 0, 255), 3)
    else:
        # Countdown finished, display "Go!" for 1 second
        if countdown_end_time is None:
            countdown_end_time = current_time
        if current_time - countdown_end_time < 1:
            cv2.putText(img, "Go!", (200, 300), cv2.FONT_HERSHEY_PLAIN, 10, (0, 255, 0), 3)
        countdown_finished = True

        # Determine the outcome of the game
        if countdown_finished:
            if player_prediction == computer_choice:
                result = "TIE"
            elif (player_prediction == "Rock" and computer_choice == "Scissors") or \
                (player_prediction == "Paper" and computer_choice == "Rock") or \
                (player_prediction == "Scissors" and computer_choice == "Paper"):
                result = "Player wins"
            else:
                result = "Computer wins"

            # Display the result on the image
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