from flask import Flask, request, render_template
import numpy as np
import joblib
import os

app = Flask(__name__)

# Load the trained model
model = joblib.load('model/model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve form inputs
    input_data = {
        'Age': float(request.form['Age']),
        'Gender': int(request.form['Gender']),
        'BMI': float(request.form['BMI']),
        'Smoker': int(request.form['Smoker']),
        'Number_of_Dependents': int(request.form['Number_of_Dependents']),
        'Type_of_Claim': int(request.form['Type_of_Claim']),
        'Claim_Amount': float(request.form['Claim_Amount']),
        'Number_of_Previous_Claims': int(request.form['Number_of_Previous_Claims']),
        'Hospital_Stay_Duration': int(request.form['Hospital_Stay_Duration']),
        'Doctor_Visits': int(request.form['Doctor_Visits'])
    }

    # Convert to array for prediction
    values = np.array(list(input_data.values())).reshape(1, -1)

    # Make prediction
    prediction = model.predict(values)[0]
    result = "✅ Genuine Claim" if prediction == 0 else "❌ Fraudulent Claim"

    return render_template('index.html',
                           prediction_text=result,
                           input_values=input_data)

if __name__ == "__main__":
    app.run(debug=True)
