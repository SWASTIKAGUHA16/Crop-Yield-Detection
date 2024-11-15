from flask import Flask, request, jsonify
import pickle
import numpy as np

# Load model and scaler
with open('dtr.pkl', 'rb') as f:
    model = pickle.load(f)

with open('preprocessor.pkl', 'rb') as f:
    scaler = pickle.load(f)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from POST request
    data = request.json
    features = np.array([data['rainfall'], data['temperature'], data['soil_quality']])

    # Scale the features
    features = scaler.transform([features])

    # Make prediction
    prediction = model.predict(features)
    return jsonify({'predicted_yield': prediction[0]})

if __name__ == '_main_':
    app.run(debug=True)