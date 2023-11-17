import streamlit as st
import pandas as pd
import numpy as np  # Added import for numpy
from joblib import load
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the scaler and model
scaler = load('/Users/nunanewton/Desktop/3377__Churning_Customers/scalerChurn1.pkl')
model = load_model('/Users/nunanewton/Desktop/3377__Churning_Customers/AImodel1.h5')

# Assume label_encoder is defined
label_encoder = LabelEncoder()

# Streamlit App
st.title("Churn Prediction App")

# Add input fields for the user to input data
st.sidebar.header("User Input")

# Add more input fields for other features...
total_charges = st.sidebar.number_input("Total Charges")
partner = st.sidebar.selectbox("Partner", ["No", "Yes"])
dependents = st.sidebar.selectbox("Dependents", ["No", "Yes"])
multiple_lines = st.sidebar.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.sidebar.selectbox("Online Security", ["No", "Yes", "No internet service"])
online_backup = st.sidebar.selectbox("Online Backup", ["No", "Yes", "No internet service"])
tech_support = st.sidebar.selectbox("Tech Support", ["No", "Yes", "No internet service"])
streaming_movies = st.sidebar.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.sidebar.selectbox("Paperless Billing", ["No", "Yes"])
payment_method = st.sidebar.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

# Create a DataFrame with the input data
input_data = pd.DataFrame({
    'Partner': [partner],
    'Dependents': [dependents],
    'MultipleLines': [multiple_lines],
    'InternetService': [internet_service],
    'OnlineSecurity': [online_security],
    'OnlineBackup': [online_backup],
    'TechSupport': [tech_support],
    'StreamingMovies': [streaming_movies],
    'Contract': [contract],
    'PaperlessBilling': [paperless_billing],
    'PaymentMethod': [payment_method],
    'TotalCharges': [total_charges]
})

# Ensure label_encoder is fitted only once
for column in input_data.columns:
    if input_data[column].dtype == 'object':
        input_data[column] = label_encoder.fit_transform(input_data[column])

# Scale the input data
scaled_input = scaler.transform(input_data)

# Make prediction
prediction = model.predict(scaled_input)
churn_probability = prediction[0][0]

st.subheader("Prediction Result")
if churn_probability > 0.5:
    st.error(f"The customer is likely to churn with a probability of {churn_probability:.2%}")
else:
    st.success(f"The customer is likely to stay with a probability of {1 - churn_probability:.2%}")
