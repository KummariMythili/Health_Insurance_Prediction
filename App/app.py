from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load Model and Scaler
model = joblib.load(os.path.join('model', 'fine_tune.pkl'))
scaler = joblib.load(os.path.join('model', 'scaler.pkl'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form
        inputs = [
            float(data['Age']),
            float(data['Gender']),
            float(data['BMI']),
            float(data['Smoker']),
            float(data['Number_of_Dependents']),
            float(data['Type_of_Claim']),
            float(data['Claim_Amount']),
            float(data['Number_of_Previous_Claims']),
            float(data['Hospital_Stay_Duration']),
            float(data['Doctor_Visits'])
        ]

        inputs_np = np.array(inputs).reshape(1, -1)
        inputs_scaled = scaler.transform(inputs_np)

        prediction = model.predict(inputs_scaled)[0]
        result = "Genuine Claim" if prediction == 0 else "Fraudulent Claim"

        return render_template('index.html', prediction_text=f'Result: {result}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
