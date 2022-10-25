import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from PIL import Image
import base64
import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np

app = Flask(__name__)
CORS(app)

def saveImg(imgData):
    imgData = base64.b64decode(imgData)
    filename = 'image.png'
    with open(filename, 'wb') as f:
        f.write(imgData)
    img = Image.open(filename)
    img.thumbnail((28, 28))
    img.save(filename)
    return filename

def getPrediction(filename):
    img = np.array(Image.open(filename))[:, :, 0]
    img = np.invert(np.array([img]))
    model = pickle.load(open('model.pkl', 'rb'))
    prediction = model.predict(img)
    return np.argmax(prediction)

@app.route('/', methods=['GET'])
def index():
    return jsonify({"App Status": "All fine here, checkout POST"})

@app.route('/', methods=['POST'])
def predict():
    print('request received')
    data = request.get_json()
    fileName = saveImg(data['imageData'])
    prediction = getPrediction(fileName)
    os.remove(fileName)
    return jsonify({'prediction': f"{prediction}"})

if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))
