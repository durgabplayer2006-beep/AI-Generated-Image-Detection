from flask import Flask, render_template, request
import os
import numpy as np
import tensorflow as tf
from PIL import Image

app = Flask(__name__)

# Load trained model
model = tf.keras.models.load_model("ai_image_detector.keras")

# Upload folder
UPLOAD_FOLDER = os.path.join(app.root_path, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    file = request.files['image']

    if file:

        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        img = Image.open(filepath).convert("RGB")
        img = img.resize((128, 128))

        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction_value = model.predict(img_array)[0][0]

        if prediction_value > 0.5:
            prediction = "REAL"
            confidence = round(prediction_value * 100, 2)
            bg_color = "green"
        else:
            prediction = "FAKE"
            confidence = round((1 - prediction_value) * 100, 2)
            bg_color = "red"

        return render_template(
            'result.html',
            prediction=prediction,
            confidence=confidence,
            image_name=file.filename,
            bg_color=bg_color
        )

    return render_template(
        'result.html',
        prediction="No Image Uploaded",
        confidence=0,
        image_name="",
        bg_color="red"
    )

if __name__ == '__main__':
    app.run(debug=True)