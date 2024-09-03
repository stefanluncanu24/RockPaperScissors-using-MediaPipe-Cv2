from PIL import Image
import numpy as np
import cv2
import mediapipe as mp
import csv
from xgboost import XGBClassifier

class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            self.mode, 
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils

        # Initialize and load the trained model
        self.model = XGBClassifier()
        self.model.load_model('xgb_model.json')

        # Variable to store the overlay image
        self.overlay_img = None

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, start_time, countdown_finished, handNo=0, draw=True):
        lmList = []
        bbox = []
        rock_paper_scissors = None  
        if self.results.multi_hand_landmarks:
            if handNo < len(self.results.multi_hand_landmarks):
                myHand = self.results.multi_hand_landmarks[handNo]
                xList = []
                yList = []
                for id, lm in enumerate(myHand.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([id, cx, cy])
                    xList.append(cx)
                    yList.append(cy)
                
                # Drawing the bounding box around the hand
                xmin, xmax = min(xList), max(xList)
                ymin, ymax = min(yList), max(yList)
                bbox = [xmin, ymin, xmax, ymax]
                if draw:
                    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                
                # Normalize the coordinates to the top-left corner of the bounding box
                normalizedLmList = [[id, cx - xmin, cy - ymin] for id, cx, cy in lmList]
                
                row_data = []
                for id, cx, cy in normalizedLmList:
                    row_data.append(cx)
                for id, cx, cy in normalizedLmList:
                    row_data.append(cy)

                # Predict the class using the trained model
                row_data = np.array(row_data).reshape(1, -1)
                prediction = self.model.predict(row_data)
                
                mapping = {0: 'Rock', 1: 'Paper', 2: 'Scissors'}

                if not countdown_finished:
                    rock_paper_scissors = mapping[prediction[0]]
    
                    with open('prediction.txt', 'w') as f:
                        f.write(rock_paper_scissors)

                    if prediction[0] == 0:
                        self.overlay_img = Image.open('rock.png').convert("RGBA")
                    elif prediction[0] == 1:
                        self.overlay_img = Image.open('paper.png').convert("RGBA")
                    elif prediction[0] == 2:
                        self.overlay_img = Image.open('scissors.png').convert("RGBA")

                    self.overlay_img = np.array(self.overlay_img)
                    self.overlay_img = cv2.cvtColor(self.overlay_img, cv2.COLOR_RGBA2BGRA)

        if self.overlay_img is not None:
            # Define the region where the overlay image will be placed
            x_offset, y_offset = 50, 150  
            overlay_width, overlay_height = 150, 150  

            overlay_img_resized = cv2.resize(self.overlay_img, (overlay_width, overlay_height))

            overlay_color = overlay_img_resized[:, :, :3]
            overlay_alpha = overlay_img_resized[:, :, 3] / 255.0

            roi = img[y_offset:y_offset+overlay_height, x_offset:x_offset+overlay_width]

            for c in range(0, 3):
                roi[:, :, c] = (1.0 - overlay_alpha) * roi[:, :, c] + overlay_alpha * overlay_color[:, :, c]

            img[y_offset:y_offset+overlay_height, x_offset:x_offset+overlay_width] = roi

        if countdown_finished:
            with open('prediction.txt', 'r') as f:
                rock_paper_scissors = f.read().strip()

        return lmList, bbox