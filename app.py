from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import base64
import os

model = tf.keras.models.load_model('model/model.h5')

app = Flask(__name__)

UPLOAD_FOLDER = "captured_images"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html', output=None)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(silent=True)

    if not data or "image" not in data:
        return jsonify({'error': 'No image data received'}), 400

    image_data = data["image"]

    if "," in image_data:
        image_data = image_data.split(",")[1]

    try: 
        image_path = os.path.join(UPLOAD_FOLDER, "captured_image.png")
        with open(image_path, "wb") as image_file:
            image_file.write(base64.b64decode(image_data))

        print(f"Image saved at: {image_path}")

        img = image.load_img(image_path, target_size=(150, 150))  
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  
        img_array = img_array / 255.0  

        prediction = model.predict(img_array)[0][0]
        prediction = np.round(prediction)

        if prediction < 0.5:
            result = '🔥🔥🔥🔥🔥🔥🔥 Fire 🔥🔥🔥🔥🔥🔥🔥'
        else:
            result = '......Not Fire......'

        print(result)

        return jsonify({'result': result})

    except Exception as e:
        print("Error:", e)
        return jsonify({'error': 'Prediction failed'}), 500

if __name__ == '__main__':
    app.run(debug=True)
