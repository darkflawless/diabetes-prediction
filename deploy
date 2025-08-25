from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Tải dữ liệu và huấn luyện mô hình
try:
    df = pd.read_csv('cleaned_diabetes.csv')
    features = ['Glucose', 'BMI', 'Age', 'DiabetesPedigreeFunction', 'Pregnancies']
    X = df[features]
    y = df['Outcome']
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
except Exception as e:
    print(f"Error loading data or training model: {e}")

@app.route('/')
def home():
    return send_file('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        # Kiểm tra dữ liệu đầu vào
        for feat in features:
            if feat not in data:
                return jsonify({"error": f"Missing field: {feat}"}), 400
        input_data = [data[feat] for feat in features]
        prediction = model.predict([input_data])[0]
        result = {"prediction": int(prediction), "message": "0 = Non-diabetic, 1 = Diabetic"}
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)