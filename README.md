# 💰 Financial Fraud Detection using Streamlit
## 🚀 Overview

This project implements a Financial Fraud Detection System built with Streamlit that leverages machine learning models to detect fraudulent transactions in real-time. It provides an interactive interface to visualize model performance and predict transaction fraud probability.

# 🎯 Objective

To build a robust and interpretable system capable of identifying fraudulent financial activities from large-scale transaction data using multiple machine learning models.

# 🧠 Models Implemented

Logistic Regression – Achieved perfect classification with ROC-AUC: 1.0000

Random Forest – Delivered high performance with ROC-AUC: 0.9999

XGBoost – Produced top-tier accuracy and recall with ROC-AUC: 0.9999

All models were evaluated using Accuracy, Precision, Recall, F1-Score, and ROC-AUC metrics.

## 📊 Model Performance Summary
| Model               | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| ------------------- | -------- | --------- | ------ | -------- | ------- |
| Logistic Regression | 0.9992   | 0.9796    | 1.0000 | 0.9897   | 1.0000  |
| Random Forest       | 0.9987   | 0.9921    | 0.9725 | 0.9822   | 0.9999  |
| XGBoost             | 0.9996   | 0.9906    | 0.9992 | 0.9949   | 0.9999  |

## 🧩 Key Features

Real-time fraud probability prediction.

Interactive Streamlit dashboard for data insights.

Visual comparison of model metrics.

Highlighted confusion matrices and ROC curves.

High interpretability through feature importance visualization.

## 🖥️ Application Preview
App Home Interface

Displays transaction upload and model selection panel.

Fraud Prediction Dashboard

Shows prediction results with fraud probability, classification outcome, and visual metrics comparison.

Performance Analytics

Interactive charts displaying accuracy, ROC-AUC, and feature importance for each model.

(Include your Streamlit app screenshots here, e.g. ![App Preview](images/app_preview.png))

## 📈 Insights

Logistic Regression achieved the highest generalization and perfect fraud recall.

XGBoost offered optimal trade-off between precision and recall.

Random Forest showed strong ensemble stability across data splits.

## 🧾 Conclusion

The system demonstrates exceptional accuracy and reliability for detecting fraudulent financial transactions. Using an intuitive Streamlit dashboard, analysts can seamlessly explore model outputs, interpret fraud risks, and make data-driven decisions in real-time.

### Author: Satish Gaikwad
### Repository: Financial-Fraud-Detection-Streamlit
### Built With: Python, Streamlit, Scikit-learn, XGBoost, RandomForest
