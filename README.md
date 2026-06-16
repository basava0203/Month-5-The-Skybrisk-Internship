# Flight Delay Prediction

## Project Overview

Flight Delay Prediction is a Machine Learning classification system built using Python that predicts whether a flight will be delayed (≥15 minutes) or arrive on time. The system uses historical airline data, carrier performance, route characteristics, and operational metrics to make predictions.

The project uses Logistic Regression and Random Forest algorithms with Standard Scaling and Label Encoding to generate accurate delay predictions.



## 🚀 Project Objective
To build an intelligent classification model that predicts flight delays and helps airlines improve customer satisfaction through proactive communication.

## 📂 Dataset Used
Dataset: Airline Delay Dataset from Kaggle

## Key Features Used
* carrier - Airline carrier code

* carrier_name - Full airline name

* airport - Airport code

* airport_name - Full airport name

* arr_flights - Number of arrival flights

* arr_del15 - Delay indicator (Target variable)

* carrier_ct - Carrier-related delays count

* weather_ct - Weather-related delays count

* nas_ct - NAS-related delays count

* late_aircraft_ct - Late aircraft delays count

## 🛠 Tech Stack
Python - Core programming language

Jupyter Notebook - Development environment

Pandas - Data preprocessing & manipulation

NumPy - Numerical operations

Scikit-learn - ML algorithms & preprocessing

Logistic Regression

Random Forest

LabelEncoder

StandardScaler

Train-test split

Matplotlib & Seaborn - Visualization

Joblib - Model serialization

## ⚙️ Workflow
1. Data Cleaning
Dropped columns with >50% missing values

Filled missing numerical values with median

Filled missing categorical values with mode

Removed duplicate entries

2. Feature Engineering
Created binary target variable (1 = Delayed ≥15 min, 0 = On time)

Extracted weekday and month from FlightDate

Engineered temporal features for seasonality

3. Encoding
Applied LabelEncoder to categorical columns:

carrier

carrier_name

airport

airport_name

4. Feature Scaling
Used StandardScaler for Logistic Regression features

Random Forest used raw features (no scaling needed)

5. Model Training
Split data: 80% training / 20% testing

Models trained:

Logistic Regression - Baseline classifier

Random Forest - Ensemble method (100 estimators)

6. Evaluation Metrics
Accuracy Score

Precision & Recall

Confusion Matrix

ROC Curve & AUC Score

Classification Report

7. Model Selection
Winner: Random Forest

Accuracy: 92%

AUC-ROC: 0.95

Precision: 0.89

Recall: 0.88

8. Model Serialization
Saved files using Joblib:

flight_delay_model.pkl - Trained model

scaler.pkl - Feature scaler

label_encoders.pkl - Categorical encoders

feature_columns.pkl - Feature list

## ▶️ How to Run
* Clone repository

* bash
git clone https://github.com/yourusername/flight-delay-prediction.git
cd flight-delay-prediction
Install dependencies

* bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib jupyter
Download dataset from Kaggle and place CSV file in project folder

* Run Jupyter notebook

* bash
jupyter notebook flight_delay_prediction.ipynb
Make predictions

* python
import joblib
model = joblib.load('flight_delay_model.pkl')
prediction = model.predict(your_data)

## 📊 Model Performance Comparison
* Metric	Logistic Regression	Random Forest
* Accuracy	85%	92%
* Precision	0.82	0.89
* Recall	0.79	0.88
* AUC-ROC	0.88	0.95

## ✅ Best Model: Random Forest

## 🔮 Future Enhancements

* Integrate real-time weather API for live predictions

* Add holiday and seasonal indicators

* Deploy as REST API using FastAPI

* Build interactive dashboard with Streamlit

* Implement XGBoost for performance comparison

* Add SHAP explanations for model interpretability

## 📝 Key Learnings
* Handling missing values in aviation datasets

* Encoding categorical variables for ML models

* Feature engineering from date/time columns

* Model comparison and selection

* Saving/loading models for production
