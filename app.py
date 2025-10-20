# fraud_app_limited_options_negtime.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model
model = joblib.load('fraud_detection_model.pkl')

# Load dataset for categorical options
df = pd.read_csv("data/fraud_data.csv")

st.title("💳 Fraud Detection App (Dynamic Options)")

st.write("Predict if a transaction is fraudulent using all features from training dataset.")

st.sidebar.header("Enter Transaction Details")

# Function to choose between dropdown or free input based on number of unique values
def dynamic_input(feature_name, df, threshold=50):
    unique_values = df[feature_name].dropna().unique()
    if len(unique_values) <= threshold:
        return st.sidebar.selectbox(feature_name.replace('_',' ').title(), unique_values)
    else:
        return st.sidebar.text_input(feature_name.replace('_',' ').title())

def user_input_features():
    # Sender/Receiver Accounts with dynamic input
    sender_account = dynamic_input("sender_account", df)
    receiver_account = dynamic_input("receiver_account", df)
    
    # Other categorical features
    transaction_type = dynamic_input("transaction_type", df)
    merchant_category = dynamic_input("merchant_category", df)
    location = dynamic_input("location", df)
    device_used = dynamic_input("device_used", df)
    payment_channel = dynamic_input("payment_channel", df)
    
    # Numeric features
    amount = st.sidebar.number_input("Transaction Amount", min_value=0.0, value=100.0)
    time_since_last_transaction = st.sidebar.number_input(
        "Time Since Last Transaction (seconds)", 
        min_value=-1e6,  # allow negative values
        max_value=1e6,
        value=3600.0
    )
    spending_deviation_score = st.sidebar.number_input("Spending Deviation Score", min_value=-1e6, value=0.5)
    velocity_score = st.sidebar.number_input("Velocity Score", min_value=0.0, value=1.0)
    geo_anomaly_score = st.sidebar.number_input("Geo Anomaly Score", min_value=0.0, value=0.2)
    hour = st.sidebar.slider("Transaction Hour (0-23)", 0, 23, 12)
    day_of_week = st.sidebar.slider("Day of Week (Monday=1, Sunday=7)", 1, 7, 3)
    sender_total_transaction = st.sidebar.number_input("Sender Total Transactions", min_value=1, value=1)
    receiver_total_transaction = st.sidebar.number_input("Receiver Total Transactions", min_value=1, value=1)
    sender_fraud_transaction = st.sidebar.number_input("Sender Fraud Transactions", min_value=0, value=0)
    receiver_fraud_transaction = st.sidebar.number_input("Receiver Fraud Transactions", min_value=0, value=0)
    
    # Engineered features
    amount_per_velocity = amount / (velocity_score + 1)
    amount_log = np.log1p(amount)
    amount_to_avg_ratio = 1
    transaction_per_day = 1
    transaction_gap = time_since_last_transaction
    is_night_transaction = int(hour >= 18)
    is_weekend = int(day_of_week in [6,7])
    is_self_transfer = int(sender_account == receiver_account)
    sender_degree = 1
    receiver_degree = 1
    sender_avg_amount = amount
    sender_std_amount = 0
    sender_fraud_percentage = (sender_fraud_transaction*100/sender_total_transaction if sender_total_transaction>0 else 0)
    receiver_fraud_percentage = (receiver_fraud_transaction*100/receiver_total_transaction if receiver_total_transaction>0 else 0)
    deviation_squared = spending_deviation_score ** 2
    
    data = {
        'sender_account': sender_account,
        'receiver_account': receiver_account,
        'transaction_type': transaction_type,
        'merchant_category': merchant_category,
        'location': location,
        'device_used': device_used,
        'payment_channel': payment_channel,
        'amount': amount,
        'time_since_last_transaction': time_since_last_transaction,
        'spending_deviation_score': spending_deviation_score,
        'velocity_score': velocity_score,
        'geo_anomaly_score': geo_anomaly_score,
        'amount_per_velocity': amount_per_velocity,
        'amount_log': amount_log,
        'amount_to_avg_ratio': amount_to_avg_ratio,
        'transaction_per_day': transaction_per_day,
        'transaction_gap': transaction_gap,
        'is_night_transaction': is_night_transaction,
        'is_weekend': is_weekend,
        'is_self_transfer': is_self_transfer,
        'sender_degree': sender_degree,
        'receiver_degree': receiver_degree,
        'sender_total_transaction': sender_total_transaction,
        'receiver_total_transaction': receiver_total_transaction,
        'sender_avg_amount': sender_avg_amount,
        'sender_std_amount': sender_std_amount,
        'sender_fraud_transaction': sender_fraud_transaction,
        'receiver_fraud_transaction': receiver_fraud_transaction,
        'sender_fraud_percentage (%)': sender_fraud_percentage,
        'receiver_fraud_percentage (%)': receiver_fraud_percentage,
        'deviation_squared': deviation_squared
    }
    
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()
st.subheader("Transaction Features Entered")
st.write(input_df)

# Prediction
if st.button("Predict Fraud"):
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)[:,1]
    
    if prediction[0] == 1:
        st.error(f"⚠️ Fraudulent Transaction! Probability: {prediction_proba[0]:.2f}")
    else:
        st.success(f"✅ Legitimate Transaction. Probability: {prediction_proba[0]:.2f}")
