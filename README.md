# 🏥 Health Insurance Claim Prediction using Machine Learning

## 📋 Project Overview

This project aims to build a machine learning-based system that predicts whether a health insurance claim is **Genuine** or **Fraudulent** based on claim details provided by users. The solution uses historical claim data, feature engineering, and advanced classification models to automate fraud detection and improve decision-making in insurance processes.

---

## 🚀 Purpose

- Automate the detection of fraudulent insurance claims.
- Save time and reduce the cost of manual investigations.
- Improve the accuracy of claim classification.
- Enhance trust and fairness in the claim settlement process.

---

## 💡 Why This Project?

Insurance fraud causes billions in losses every year and increases premiums for honest policyholders. Traditional manual detection methods are slow, costly, and inefficient. This project demonstrates how **machine learning** can address these issues through an intelligent, automated, and scalable solution.

---

## 🛠 Methodology

1. **Data Collection & Preprocessing**
   - Used a balanced dataset of genuine and fraudulent claims.
   - Performed label encoding for categorical variables.
   - Scaled numerical features.
   - Conducted feature engineering to boost model performance.

2. **Model Training**
   - Trained multiple algorithms:
     - Logistic Regression
     - Support Vector Machine (SVM)
     - Random Forest Classifier
     - AdaBoost Classifier
     - Gradient Boosting Classifier

3. **Model Selection**
   - Chose **Gradient Boosting Classifier** for its superior accuracy and robustness.

4. **Evaluation & Fine-Tuning**
   - Performed hyperparameter tuning for optimal performance.
   - Validated using cross-validation.

5. **Deployment**
   - Built a **Flask web application** for real-time predictions.
   - Designed an intuitive **HTML/CSS/JavaScript frontend**.

---

## 🤖 Model Used & Justification

**Gradient Boosting Classifier** was selected because:
- It delivered the highest accuracy on the test set.
- It handles non-linear patterns and feature interactions well.
- It is resistant to overfitting when properly tuned.
- Ideal for both small and medium-sized datasets.

---

## ⚙️ Requirements

### Software:
- Python 3.x
- scikit-learn
- pandas, numpy
- Flask
- HTML, CSS, JavaScript

### Hardware:
- Minimum: 4GB RAM, any modern processor
- Recommended: 8GB RAM for faster training

---

## 🌐 Project Structure

Project/
├── App/
│ ├── app.py
│ ├── model/
│ │ └── model.pkl
│ ├── templates/
│ │ └── index.html
│ ├── static/
│ │ ├── main_css.css
│ │ └── main.js
├── Data/
│ ├── balanced_health_insurance_claims.csv
│ └── preprocessed_data.csv
├── Training/
│ ├── preprocess_data.ipynb
│ └── training_notebook.ipynb
├── Evaluation/
│ ├── best_model_saving.ipynb
│ └── evaluation_and_tuning.ipynb
├── requirements.txt
└── README.md

---

## 🎯 How to Run

1. Install the required libraries:

2. Run preprocessing and training notebooks.

3. Launch the web app:

4. Open your browser and go to `http://localhost:5000` to use the predictor.

---

## ✅ Conclusion

This project showcases how machine learning can automate and improve fraud detection in health insurance claims. The deployed solution helps insurance companies save time, reduce financial losses, and process claims more fairly and efficiently.

---

