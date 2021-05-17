import base64
import io
import numpy as np

from PIL import Image
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image


model_path = "models/resnet50/resnet50_trunk12_74_0.9250.h5"
img_target_size = (224, 224)
tree_types = ['Alder', 'Beech', 'Birch', 'Chestnut', 'Ginkgo_biloba', 'Hornbeam', 'Horse_chestnut',
             'Linden', 'Oak', 'Oriental_plane', 'Pine', 'Spruce']

app = Flask(__name__)


def get_model():
    global model
    model = load_model(model_path)
    print(" * Model loaded! ")


def preprocess_image(img):
    img = img.resize(img_target_size)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img) / 255

    return img


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/form', methods=['POST', 'GET'])
def form():
    if request.method == 'POST':
        msg = request.get_json(force=True)
        name = msg['name']
        response = {
            'greeting': 'Hello, ' + name + '!'
        }
        return jsonify(response)
    else:
        return render_template('form.html')


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        msg = request.get_json(force=True)
        print(" * preprocessing image... ")
        encoded = msg['image']
        decoded = base64.b64decode(encoded)
        img = Image.open(io.BytesIO(decoded))
        processed_image = preprocess_image(img)
        print(" * image preprocessed ")

        print(" * get prediction... ")
        prediction = model.predict(processed_image).tolist()

        response = {'prediction': {tree_types[i] : prediction[0][i] for i in range(len(prediction[0]))}}
        print(" * sending results")
        return jsonify(response)
    else:
        return render_template('predict.html')


if __name__ == '__main__':
    print(" * Loading Keras model...")
    get_model()

    app.run(debug=True, host='0.0.0.0')