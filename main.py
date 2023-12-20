import os
import pandas as pd
import numpy as np
from flask import Flask, jsonify, request
import pickle as pkl
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import random, re, json, string

app = Flask(__name__)
app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg'])
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['MODEL_FILE'] = 'model_chatbot.h5'
app.config['TOKENIZER'] = 'tokenizer.pkl'
app.config['LABEL_ENCODER'] = 'label_encoder.pkl'
app.config['RESPONSE'] = 'responses.json'


model = load_model(app.config['MODEL_FILE'], compile=False)
tokenizer = pkl.load(open(app.config['TOKENIZER'], "rb"))
label_encoder = pkl.load(open(app.config['LABEL_ENCODER'], "rb"))
response = json.load(open(app.config['RESPONSE']))
# tag_dict = {}
# with open(app.config['LABELS_FILE'], 'r') as file:
#     labels = file.read().splitlines()


def predict_sentence(sentence):

    texts_p = []

    prediction_input = [letters.lower() for letters in sentence if letters not in string.punctuation]
    prediction_input = ''.join (prediction_input)
    texts_p.append(prediction_input)

    prediction_input= tokenizer.texts_to_sequences(texts_p)
    prediction_input=np.array(prediction_input).reshape(-1)
    prediction_input = pad_sequences([prediction_input], 13)

    output = model.predict(prediction_input)
    output = output.argmax()
    response_tag = label_encoder.inverse_transform([output])[0]
    respon = random.choice(response[response_tag])

    return response_tag, respon


@app.route("/")
def index():
    return "Hello World!"


@app.route("/prediction", methods=["POST"])
def prediction_route():
    if request.method == "POST":
        sentences = str(request.json['text'])
        if sentences is not None:
            response_tag, response = predict_sentence(sentences)

            return jsonify({
                "status": {
                    "code": 200,
                    "message": "Success predicting"
                },
                "data": {
                    "class": response_tag,
                    "response": response,
                }
            }), 200
        else:
            return jsonify({
                "status": {
                    "code": 400,
                    "message": "Text should not be none."
                },
                "data": None,
            }), 400
    else:
        return jsonify({
            "status": {
                "code": 405,
                "message": "Method not allowed"
            },
            "data": None,
        }), 405


if __name__ == "__main__":
    app.run(debug=True,
            host="0.0.0.0",
            port=int(os.environ.get("PORT", 8080)))