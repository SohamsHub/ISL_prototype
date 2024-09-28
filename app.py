from flask import Flask, render_template, Response, jsonify, make_response, request
import cv2
import numpy as np
import tensorflow as tf
from collections import deque
from tensorflow.keras.models import load_model
import mediapipe as mp
import io

# tf.compat.v1.disable_eager_execution()

# Custom class for gesture recognition
class SignLanguageRecognizer:
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.holistic = self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        self.model = load_model('sign_language_model.h5')
        self.max_length = 90
        self.gesture_labels = ['Alive', 'Bad', 'Beautiful', 'Big large', 'Blind', 'Cheap', 'Clean', 'Cold', 'Cool', 'Curved', 
                               'Dead', 'Deaf', 'Deep', 'Dirty', 'Dry', 'Expensive', 'Famous', 'Fast', 'Female', 'Flat', 'Good', 
                               'Happy', 'Hard', 'Healthy', 'Heavy', 'High', 'Hot', 'Light', 'Long', 'Loose', 'Loud', 'Low', 
                               'Male', 'Mean', 'Narrow', 'New', 'Nice', 'Old', 'Poor', 'Quiet', 'Rich', 'Sad', 'Shallow', 
                               'Short', 'Sick', 'Slow', 'Small little', 'Soft', 'Strong', 'Tall', 'Thick', 'Thin', 'Tight', 
                               'Ugly', 'Warm', 'Weak', 'Wet', 'Wide', 'Young']
        
        self.sequence = deque(maxlen=self.max_length)
        self.predictions = []

    def extract_landmarks(self, frame):
        results = self.holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
        face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
        
        return np.concatenate([lh, rh, pose, face])

    def recognize_gesture(self):
        if len(self.sequence) == self.max_length:
            padded_sequence = np.array(self.sequence)
            res = self.model.predict(np.expand_dims(padded_sequence, axis=0))[0]
            self.predictions.append(np.argmax(res))

            if len(self.predictions) > 5:
                self.predictions = self.predictions[-5:]

            if len(self.predictions) >= 3 and len(set(self.predictions[-3:])) == 1:
                return self.gesture_labels[self.predictions[-1]]
            return ""
        return ""

    def process_frame(self, frame):
        
        landmarks = self.extract_landmarks(frame)
        gesture = self.recognize_gesture()
        self.sequence.append(landmarks)

        if len(self.sequence) > 90:  # You can adjust this limit as needed
            self.sequence.pop(0) 

        return gesture

# Flask application setup
app = Flask(__name__)

# Initialize the recognizer
recognizer = SignLanguageRecognizer()

# CSS content
CSS_CONTENT = """
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: Arial, sans-serif;
    background-color: #000;
    color: #fff;
    text-align: center;
    padding: 20px;
}

.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
    background-color: #222;
    border-radius: 8px;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.5);
}

h1 {
    font-size: 2.5em;
    color: #fff;
    margin-bottom: 20px;
}

p {
    font-size: 1.2em;
    margin-bottom: 20px;
}

img {
    border: 2px solid #444;
    border-radius: 8px;
}

#gesture-labels {
    margin-top: 20px;
}

#gesture-labels h2 {
    font-size: 1.8em;
    margin-bottom: 15px;
    color: #ccc;
}

#gesture-labels ul {
    list-style-type: none;
    padding: 0;
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
}

#gesture-labels li {
    padding: 8px 12px;
    border: 1px solid #444;
    margin: 5px;
    border-radius: 4px;
    background-color: #333;
}
"""

@app.route('/')
def index():
    return render_template('index.html', labels=recognizer.gesture_labels)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/styles.css')
def styles():
    response = make_response(CSS_CONTENT)
    response.headers['Content-Type'] = 'text/css'
    return response

@app.route('/gesture_labels')
def gesture_labels():
    return jsonify(recognizer.gesture_labels)

@app.route('/process_frame/', methods=['POST'])
@app.route('/process_frame', methods=['POST'])
def process_frame():
    print(request)
    if 'frame' not in request.files:
        return jsonify({'error': 'No frame part'}), 400
    
    frame_file = request.files['frame']
    if frame_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Read the image file and process it
    image_bytes = frame_file.read()
    image = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(image, cv2.IMREAD_COLOR)
    
    # Process the frame
    gesture = recognizer.process_frame(frame)
    
    return jsonify({'gesture': gesture})

# Run the application
if __name__ == "__main__":
    app.run(debug=True)