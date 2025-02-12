from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import cv2
import os

# Initialize Flask app
app = Flask(__name__)

# Load the Haarcascade and model
facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load the pre-trained model

def FacialExpressionModel(json_file, weights_file):
    with open(json_file, "r") as file:
        loaded_model_json = file.read()
    model = model_from_json(loaded_model_json)
    model.load_weights(weights_file)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = FacialExpressionModel("model_a1.json", "model_weights1.h5")

EMOTIONS_LIST = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Helper function to predict emotion
def detect_emotion(file_path):
    image = cv2.imread(file_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = facec.detectMultiScale(gray_image, 1.3, 5)
    try:
        for (x, y, w, h) in faces:
            fc = gray_image[y:y+h, x:x+w]
            roi = cv2.resize(fc, (48, 48))
            roi = roi[np.newaxis, :, :, np.newaxis]
            pred = EMOTIONS_LIST[np.argmax(model.predict(roi))]
            return pred
    except Exception as e:
        return "Unable to detect"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(url_for('home'))

    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('home'))

    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Predict emotion
        emotion = detect_emotion(filepath)
        return render_template('result.html', emotion=emotion, image_path=filepath)

if __name__ == '__main__':
    app.run(debug=True)

# HTML files structure:
# templates/
#   - index.html
#   - result.html
