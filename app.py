from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open("calorie_model.pickle", "rb") as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from JSON
    data = request.json
    gender = int(data.get("gender", 0))  # 1 for male, 0 for female
    age = float(data.get("age", 0))
    height = float(data.get("height", 0))
    weight = float(data.get("weight", 0))
    duration = float(data.get("duration", 0))
    heart_rate = float(data.get("heart_rate", 0))
    body_temp = float(data.get("body_temp", 0))
    
    # Create input array for prediction
    features = np.array([[gender, age, height, weight, duration, heart_rate, body_temp]])
    prediction = model.predict(features)
    
    # Convert numpy.float32 to Python float
    result = float(prediction[0])
    
    return jsonify({'calories_burnt': result})

if __name__ == '__main__':
    app.run(debug=True)