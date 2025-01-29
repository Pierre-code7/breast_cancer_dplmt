from flask import Flask, request, jsonify
import joblib
import pandas as pd
from prometheus_client import Counter, generate_latest

app = Flask(__name__)

# Load model
model = joblib.load("breast_cancer_model.pkl")

@app.route('/')
def home():
    return "Welcome to the ML API. Use the /predict endpoint to make predictions."

@app.route('/favicon.ico')
def favicon():
    return "", 204
api_requests = Counter("api_requests", "Number of API requests")

@app.before_request
def before_request():
    api_requests.inc()

#@app.route('/predict', methods=['POST'])
#def predict():
#    data = request.json
#    df = pd.DataFrame(data)
#    predictions = model.predict(df)
#    return jsonify({"predictions": predictions.tolist()})

@app.route('/metrics')
def metrics():
    return generate_latest(), 200
    
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json  # Simulating request handling
        # Your prediction logic here
        return {"result": "ok"}, 200
    except Exception:
        error_counter.inc()
        return {"error": "prediction failed"}, 500




if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
