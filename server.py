import tensorflow as tf
import cv2
from PIL import Image
import base64
import pickle
from flask import Flask, request, jsonify
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


app = Flask(__name__)


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
    img = cv2.imread(filename)[:, :, 0]
    img = np.invert(np.array([img]))
    model = pickle.load(open('model.pkl', 'rb'))
    prediction = model.predict(img)
    return np.argmax(prediction)


@app.route('/', methods=['POST'])
def predict():
    print('request received')
    data = request.get_json()
    fileName = saveImg(data['imageData'])
    prediction = getPrediction(fileName)
    os.remove(fileName)
    return jsonify({'prediction': f"{prediction}"})


if __name__ == '__main__':
    app.run(port=5000, debug=True)
