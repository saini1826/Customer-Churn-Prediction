import streamlit as st
import numpy as np
import pandas as pd
import joblib
import pickle
import base64


# Page Configuration

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon=" ",
    layout="wide"
)


encoders = joblib.load("encoders.pkl")
model = joblib.load("model.pkl")



def add_banner():
    with open("banner_3.jpg", "rb") as image:
        encoded = base64.b64encode(image.read()).decode()

    st.markdown(
        f"""
        <style>
        .banner {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            height: 200px;
            border-radius: 15px;
            display: flex;
            justify-content: center;
            align-items: center;
            margin-bottom: 30px;
        }}

        .banner h1 {{
            color: white;
            font-size: 70px;
            font-weight: bold;
            background: rgba(0,0,0,0.55);
            padding: 15px 30px;
            border-radius: 35px;
            text-shadow: 2px 2px 8px black;
        }}
        </style>

        <div class="banner">
            <h1> Customer churn prediction </h1>
        </div>
        """,
        unsafe_allow_html=True
    )

add_banner()

st.write("Predict whether a customer is likely to churn or not.")

st.sidebar.header("Customer Information")

gender = st.sidebar.selectbox(
    "Gender",
    ["Female", "Male"]
)

senior_citizen = st.sidebar.selectbox(
    "Senior Citizen",
    [0, 1]
)

partner = st.sidebar.selectbox(
    "Partner",
    ["No", "Yes"]
)

dependents = st.sidebar.selectbox(
    "Dependents",
    ["No", "Yes"]
)

tenure = st.sidebar.number_input(
    "Tenure (Months)",
    min_value=0,
    max_value=100,
    value=12
)

phone_service = st.sidebar.selectbox(
    "Phone Service",
    ["No", "Yes"]
)

multiple_lines = st.sidebar.selectbox(
    "Multiple Lines",
    ["No", "No phone service", "Yes"]
)

internet_service = st.sidebar.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

online_security = st.sidebar.selectbox(
    "Online Security",
    ["No", "No internet service", "Yes"]
)

online_backup = st.sidebar.selectbox(
    "Online Backup",
    ["No", "No internet service", "Yes"]
)

device_protection = st.sidebar.selectbox(
    "Device Protection",
    ["No", "No internet service", "Yes"]
)

tech_support = st.sidebar.selectbox(
    "Tech Support",
    ["No", "No internet service", "Yes"]
)

streaming_tv = st.sidebar.selectbox(
    "Streaming TV",
    ["No", "No internet service", "Yes"]
)

streaming_movies = st.sidebar.selectbox(
    "Streaming Movies",
    ["No", "No internet service", "Yes"]
)

contract = st.sidebar.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)

paperless_billing = st.sidebar.selectbox(
    "Paperless Billing",
    ["No", "Yes"]
)

payment_method = st.sidebar.selectbox(
    "Payment Method",
    [
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check"
    ]
)

monthly_charges = st.sidebar.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=70.0
)

total_charges = st.sidebar.number_input(
    "Total Charges",
    min_value=0.0,
    value=1000.0
)


# Label Encoding
# Same alphabetical order as sklearn LabelEncoder


gender_map = {
    "Female": 0,
    "Male": 1
}

yes_no_map = {
    "No": 0,
    "Yes": 1
}

multiple_lines_map = {
    "No": 0,
    "No phone service": 1,
    "Yes": 2
}

internet_service_map = {
    "DSL": 0,
    "Fiber optic": 1,
    "No": 2
}

internet_features_map = {
    "No": 0,
    "No internet service": 1,
    "Yes": 2
}

contract_map = {
    "Month-to-month": 0,
    "One year": 1,
    "Two year": 2
}

payment_method_map = {
    "Bank transfer (automatic)": 0,
    "Credit card (automatic)": 1,
    "Electronic check": 2,
    "Mailed check": 3
}

st.sidebar.image("banner_2.jpg")
st.sidebar.title("About Project")
st.sidebar.write("Customer Churn Prediction is a Machine Learning project "
    " that predicts whether a customer is likely to leave a service "
    "based on customer demographics, account information and "
    "service-related features")

st.sidebar.title("Project Objective 🎯")
st.sidebar.write("Identify customers who are likely to churn and help businesses "
    "take proactive customer retention actions.")

st.sidebar.title("Contact me 📱")
st.sidebar.write("For project-related queries, connect through Email or GitHub")
st.sidebar.write("abhisheksaini80610@gmail.com")

st.sidebar.image("github.png")
st.sidebar.write("https://github.com/saini1826")

st.sidebar.title("About us 👥")
st.sidebar.write("Developed by Abhishek saini ")

# Convert Inputs

input_data = {
    "gender": gender_map[gender],
    "SeniorCitizen": senior_citizen,
    "Partner": yes_no_map[partner],
    "Dependents": yes_no_map[dependents],
    "tenure": tenure,
    "PhoneService": yes_no_map[phone_service],
    "MultipleLines": multiple_lines_map[multiple_lines],
    "InternetService": internet_service_map[internet_service],
    "OnlineSecurity": internet_features_map[online_security],
    "OnlineBackup": internet_features_map[online_backup],
    "DeviceProtection": internet_features_map[device_protection],
    "TechSupport": internet_features_map[tech_support],
    "StreamingTV": internet_features_map[streaming_tv],
    "StreamingMovies": internet_features_map[streaming_movies],
    "Contract": contract_map[contract],
    "PaperlessBilling": yes_no_map[paperless_billing],
    "PaymentMethod": payment_method_map[payment_method],
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges
}


# Create DataFrame

input_df = pd.DataFrame([input_data])


# Show Input Data

st.subheader("Customer Details")

st.dataframe(
    input_df,
    use_container_width=True
)

# Prediction

st.divider()

st.markdown("""
<style>
div.stButton > button {
    background-color: #28A745;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 12px 25px;
    font-size: 18px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

if st.button("🔮 Predict Churn", use_container_width=True):

    prediction = model.predict(input_df)[0]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ Customer is likely to CHURN.")

    else:
        st.success("✅ Customer is NOT likely to CHURN.")


    # Prediction Probability
    

    if hasattr(model, "predict_proba"):

        probability = model.predict_proba(input_df)[0]

        churn_probability = probability[1] * 100
        not_churn_probability = probability[0] * 100

        st.write(
            f"**Churn Probability:** {churn_probability:.2f}%"
        )

        st.write(
            f"**Not Churn Probability:** {not_churn_probability:.2f}%"
        )

        st.progress(int(churn_probability))


# Footer

st.markdown("---")

st.caption(
    "Customer Churn Prediction | Machine Learning Project"
)