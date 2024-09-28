# NMF of ISL Detection - An Indian ☝ Sign 👋 Language 🖐️ app.
[![Site](https://img.shields.io/static/v1?label=visit%20Website&message=ISL%20Detection%20Page&color=yellow)]()
[![python](https://img.shields.io/static/v1?label=Python&message=%3E=3.9&color=orange&logo=python)]()
[![OpenCV](https://img.shields.io/static/v1?label=OpenCV-headless-python&message=4.8.1.78&color=blue&logo=opencv)]()
[![Tensorflow](https://img.shields.io/static/v1?label=Tensorflow&message=2.14.0&color=yellow&logo=tensorflow)]()
[![mediapipe](https://img.shields.io/static/v1?label=Mediapipe&message=0.10.7&color=green&logo=mediapipe)]()
[![Django](https://img.shields.io/badge/Python-Django-green)]()
<br>

# Objective
The objective of this project is to develop a system that accurately captures and interprets non-manual features of Indian Sign Language (ISL), such as facial expressions, head movements, and body gestures, and converts them into text. This will enhance the automatic translation of ISL by incorporating the crucial non-manual elements that convey grammatical structure, emphasis, and emotions, ensuring more precise and meaningful communication.

# About the App
#### This Web App provides:
The Django app for capturing non-manual features of Indian Sign Language (ISL) is designed to process video input, detect facial expressions, head movements, and body gestures, and convert these non-manual features into text. The app utilizes machine learning models integrated with Django’s backend to handle data processing and feature recognition. Users can upload video content, and the app will analyze the non-manual cues in ISL, translating them into meaningful text output. This app aims to enhance accessibility and improve the accuracy of ISL translation by incorporating essential non-manual elements.
<br>
<img src = 'images/output.jpeg' width='600' height='600'>
</details>

# Data Analysis
<details>
    <summary></summary>

# Source Data
(https://zenodo.org/records/4010759)


# Preprocessing
- Extract Landmarks: Use MediaPipe's Holistic model or pose detection model to extract key landmarks from the frames of a video.
- Normalization: Normalize the extracted landmark coordinates.
- Augmentation (Optional): Apply data augmentation techniques to increase diversity in your training data.
- Saving Landmarks: Store the processed landmarks in a structured format such as .npy files.


## Installation
To set up and use Sentiment Analyzer in your own project, follow these steps:

- Clone this repository to your local machine.
```
    https://github.com/mohar69/gesture_recognition.git
```
- Install the required dependencies for python 3.9 or above.
```
    pip install -r requirements.txt
```
- Running the Django app
```
    python manage.py runserver
```