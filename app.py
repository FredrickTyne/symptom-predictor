import os
from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
import pandas as pd

app = Flask(__name__)

# 加载模型与工具
model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# 特征列顺序，必须与你训练时的 X.columns 顺序一致
feature_columns = [
    'Age', 'Gender', 'Temperature (C)', 'Humidity', 'Wind Speed (km/h)',
    'nausea', 'joint_pain', 'abdominal_pain', 'high_fever', 'chills',
    'fatigue', 'runny_nose', 'pain_behind_the_eyes', 'dizziness', 'headache',
    'chest_pain', 'vomiting', 'cough', 'shivering', 'asthma_history',
    'high_cholesterol', 'diabetes', 'obesity', 'hiv_aids', 'nasal_polyps',
    'asthma', 'high_blood_pressure', 'severe_headache', 'weakness',
    'trouble_seeing', 'fever', 'body_aches', 'sore_throat', 'sneezing',
    'diarrhea', 'rapid_breathing', 'rapid_heart_rate', 'pain_behind_eyes',
    'swollen_glands', 'rashes', 'sinus_headache', 'facial_pain',
    'shortness_of_breath', 'reduced_smell_and_taste', 'skin_irritation',
    'itchiness', 'throbbing_headache', 'confusion', 'back_pain', 'knee_ache'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = []
        for col in feature_columns:
            val = request.form.get(col)
            input_data.append(float(val))

        # 转换为 DataFrame，便于处理
        df_input = pd.DataFrame([input_data], columns=feature_columns)

        # 对数值列缩放
        numeric_cols = ['Temperature (C)', 'Humidity', 'Wind Speed (km/h)']
        df_input[numeric_cols] = scaler.transform(df_input[numeric_cols])

        # 预测
        prediction = model.predict(df_input)[0]
        predicted_label = label_encoder.inverse_transform([prediction])[0]

        return jsonify({'prediction': predicted_label})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
