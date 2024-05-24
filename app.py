from flask import Flask, request, jsonify
import pickle
import numpy as np
import sklearn
import requests
import io

# utwórz obiekt app
app = Flask(__name__)

@app.route("/api/predict", methods=['GET'])
def fun():
    url = "https://github.com/michalparuch/perceptronapp/raw/main/perceptron.pkl"
    response = requests.get(url)
    response.raise_for_status()  # Ensure the request was successful

    picklefile = io.BytesIO(response.content)
    nn = pickle.load(picklefile)
    x1 = request.args.get("x1", 0, type=float)
    x2 = request.args.get("x2", 0, type=float)
    features = {'x1': x1, 'x2': x2}
    predicted_class = nn.predict([[x1, x2]])[0]
    return jsonify(features=features, predicted_class=predicted_class)


if __name__ == '__main__':
    app.run()
