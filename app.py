import streamlit as st
import pandas as pd
import joblib
import pickle

# Load the Model and Feature Columns
@st.cache_resource
def load_model_and_features():
    model = joblib.load('xgboost_churn_model.pkl')
    with open('model_features.pkl', 'rb') as f:
        features = pickle.load(f)
    return model, features

model, expected_features = load_model_and_features()

# Build the Frontend UI
st.set_page_config(page_title="Churn Predictor", page_icon="📉", layout="centered")

st.title("Telecom Customer Churn Predictor")
st.write("Enter the customer's details below to predict their likelihood of canceling their service.")

# Create columns for better layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Customer Demographics")
    senior_citizen = st.selectbox("Is the customer a Senior Citizen?", ["No", "Yes"])
    partner = st.selectbox("Do they have a Partner?", ["No", "Yes"])
    dependents = st.selectbox("Do they have Dependents?", ["No", "Yes"])
    gender = st.selectbox("Gender", ["Female", "Male"])

with col2:
    st.subheader("Account Details")
    tenure = st.slider("Tenure (Months)", 0, 72, 12)
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=15.0, max_value=120.0, value=50.0)
    total_charges = st.number_input("Total Charges ($)", min_value=15.0, max_value=9000.0, value=600.0)

st.subheader("Services Subscribed")
col3, col4 = st.columns(2)
with col3:
    phone_service = st.selectbox("Phone Service?", ["No", "Yes"])
    multiple_lines = st.selectbox("Multiple Lines?", ["No", "Yes"])
    internet_service = st.selectbox("Internet Service Type", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security?", ["No", "Yes"])
with col4:
    device_protection = st.selectbox("Device Protection?", ["No", "Yes"])
    tech_support = st.selectbox("Tech Support?", ["No", "Yes"])
    streaming_tv = st.selectbox("Streaming TV?", ["No", "Yes"])
    streaming_movies = st.selectbox("Streaming Movies?", ["No", "Yes"])

st.subheader("Billing Information")
paperless_billing = st.selectbox("Paperless Billing?", ["No", "Yes"])
payment_method = st.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])

# Process the Input Data
if st.button("Predict Churn Risk"):
    # Determine Tenure Group based on logic used in training
    if tenure <= 12:
        tenure_group = "0-1 yr"
    elif tenure <= 24:
        tenure_group = "1-2 yrs"
    elif tenure <= 48:
        tenure_group = "2-4 yrs"
    elif tenure <= 60:
        tenure_group = "4-5 yrs"
    else:
        tenure_group = "5-6 yrs"

    # Create a raw dictionary from inputs
    input_dict = {
        "gender": gender,
        "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": "No",
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Tenure_Group": tenure_group
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([input_dict])
    
    # Apply get_dummies
    input_encoded = pd.get_dummies(input_df, drop_first=True, dtype=int)
    
    # Ensure all expected columns are present, fill missing ones with 0
    # This prevents errors if a user selects an option that drops a dummy column
    final_input = pd.DataFrame(columns=expected_features)
    for col in expected_features:
        if col in input_encoded.columns:
            final_input[col] = input_encoded[col]
        else:
            final_input[col] = 0

    # Make Prediction
    prediction = model.predict(final_input)[0]
    probability = model.predict_proba(final_input)[0][1] * 100

    st.markdown("---")
    if prediction == 1:
        st.error(f"**High Risk of Churn!** (Probability: {probability:.1f}%)")
        st.write("Recommendation: Consider offering a retention discount or contacting them for feedback.")
    else:
        st.success(f"**Customer is likely to stay.** (Probability of leaving: {probability:.1f}%)")
