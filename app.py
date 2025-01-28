from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load model
model = joblib.load("breast_cancer_model.pkl")

@app.route('/')
def home():
    return "Welcome to the ML API. Use the /predict endpoint to make predictions."

@app.route('/favicon.ico')
def favicon():
    return "", 204

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame(data)
    predictions = model.predict(df)
    return jsonify({"predictions": predictions.tolist()})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
