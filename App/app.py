from flask import Flask, request, render_template
import numpy as np
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('model/model.pkl')

# If using a scaler, load it too
scaler = joblib.load('model/scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form inputs
    input_data = {
        'age': float(request.form['age']),
        'gender': int(request.form['gender']),
        'bmi': float(request.form['bmi']),
        'smoker': int(request.form['smoker']),
        'dependents': int(request.form['dependents']),
        'claim_type': int(request.form['claim_type']),
        'claim_amount': float(request.form['claim_amount']),
        'previous_claims': int(request.form['previous_claims']),
        'stay_duration': int(request.form['stay_duration']),
        'doctor_visits': int(request.form['doctor_visits'])
    }

    # Prepare data for prediction
    input_array = np.array(list(input_data.values())).reshape(1, -1)

    # Scale input data
    input_scaled = scaler.transform(input_array)

    # Make prediction
    prediction = model.predict(input_scaled)[0]
    result = "✅ Genuine Claim" if prediction == 0 else "❌ Fraudulent Claim"

    return render_template('index.html',
                           prediction_text=result,
                           request=request)

if __name__ == '__main__':
    app.run(debug=True)
