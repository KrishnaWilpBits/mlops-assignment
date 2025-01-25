from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load("model.pkl")

@app.route('/test', methods=["GET"])
def test():
    return 'hello'

@app.route("/predict", methods=["POST"])
def predict():
    # Get input data from the request
    data = request.json
    features = np.array(data["features"]).reshape(1, -1)

    # Make a prediction
    prediction = model.predict(features)
    return jsonify({"prediction": int(prediction[0])})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)