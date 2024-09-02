# OpenCV-MediaPipe-RPS

## Overview
OpenCV-MediaPipe-RPS is an interactive Rock-Paper-Scissors game that utilizes advanced hand tracking technology with OpenCV and MediaPipe to recognize hand gestures in real-time. This project leverages machine learning to interpret the player's hand gestures as rock, paper, or scissors, enabling a fun and engaging way to play the classic game against a computer.

<p align="center">
  <img src="https://github.com/user-attachments/assets/ef505285-7f85-4021-a3a7-296f22457613" alt="Image 1" width="30%">
  <img src="https://github.com/user-attachments/assets/c1a49b7e-d43e-4c01-975e-655b80b3edda" alt="Image 2" width="30%">
  <img src="https://github.com/user-attachments/assets/ef0e9fa7-cc35-48ce-ab16-bbdc65d61556" alt="Image 3" width="30%">
</p>


## Features
- Real-time hand gesture recognition.
- Play Rock-Paper-Scissors against the computer.
- Visual feedback of both player's and computer's choices.
- Countdown timer for gesture preparation.

## Installation
To run this project, you will need Python installed on your system along with the following libraries:
- OpenCV
- MediaPipe
- NumPy
- PIL
- XGBoost

## Files Description
- MyNewGameHandTracking.py: This script contains the main game logic including video capture, hand detection, gesture recognition, and game outcome determination.

- HandTrackingModule.py: A module for hand detection and gesture recognition, which predicts the player's hand gesture and displays it on the screen.

- model.py: Contains the training routines for the gesture recognition model and saves it for later use in the game.

- rps.txt: The dataset used for training the gesture recognition model, consisting of various hand positions labeled as rock, paper, or scissors.

## How It Works
1. Hand Detection: Using MediaPipe, the game detects the presence of a hand and identifies key landmarks.
2. Gesture Recognition: The detected hand gesture is compared against a trained XGBoost model to classify it as rock, paper, or scissors.
3. Game Logic: The game logic compares the player's gesture against a randomly selected choice by the computer to determine the outcome of the game.
4. Display Results: Results are displayed in real-time on the screen with the choices and outcome of each round.

## Contributions
Contributions are welcome! If you have improvements or bug fixes, please feel free to fork the repository and submit a pull request.
