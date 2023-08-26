
# Real-time Face Recognition System using LBPH Algorithm
![Screenshot 2023-08-26 181353](https://github.com/Ymanawat/FaceRecognizer/assets/81252768/6777967c-286c-47ae-9365-07228b8a10ec)

## Introduction
Welcome to the Real-time Face Recognition System using the Local Binary Patterns Histograms (LBPH) algorithm. This repository showcases an approach to face detection and identification using local texture patterns. By combining OpenCV and Python, this project offers a practical demonstration of computer vision techniques.

## Prerequisites
Make sure you have a functional Python environment installed.
Install the required packages using the following command:

`pip install opencv-python`

## Usage
1. Clone or download this repository onto your local machine.
2. Navigate to the project directory.
3. Execute the main script using this command: `python main.py`

Alternatively, you can directly launch and run `main.py` through a Python interpreter.

## Understanding the LBPH Algorithm
LBPH (Local Binary Patterns Histograms) is a well known algorithm for face recognition:
- **Pixel Labeling:** LBPH involves comparing pixel intensities with their local neighbors, resulting in binary patterns.
- **Parameters Used:** The algorithm's effectiveness lies in four crucial parameters: 
  - **Radius:** Imagine drawing a circle around a dot - this circle's size is set by the radius. 
  - **Neighbors:** It's like asking friends for their opinions - we ask neighboring dots for their colors.
  - **Grid X:** Imagine a checkerboard - this tells the computer how many squares(pixels) to make from left to right.
  - **Grid Y:** It says how many squares(pixels) to make from top to bottom.
- **Training Phase:** The model learns from labeled facial images during the training process.
- **Recognition Phase:** For recognition, the model evaluates input patterns against learned ones and assigns corresponding labels.

## Screenshots
![Screenshot 2023-08-26 181416](https://github.com/Ymanawat/FaceRecognizer/assets/81252768/7681881b-ef7c-4cec-84b4-ae1bd329f53e) ![Screenshot 2023-08-26 200339](https://github.com/Ymanawat/FaceRecognizer/assets/81252768/04252798-4560-4e1d-a60e-964813038499) ![Screenshot 2023-08-26 200449](https://github.com/Ymanawat/FaceRecognizer/assets/81252768/54e6f77c-6b2d-4527-81cc-892f371b3a82) ![Screenshot 2023-08-26 200424](https://github.com/Ymanawat/FaceRecognizer/assets/81252768/4b5133b4-cf32-4fb6-8c60-3744359cbfb9)
![Screenshot 2023-08-26 181353](https://github.com/Ymanawat/FaceRecognizer/assets/81252768/1f320ea6-7ee4-48b1-8528-231a8c3eb825)

