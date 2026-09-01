import streamlit as st
import requests


st.set_page_config(
    page_title="CustomerPulse AI",
    page_icon="📊",
    layout="wide"
)


st.title(
    "🚀 CustomerPulse AI"
)

st.subheader(
    "AI-Powered Customer Churn Prediction"
)


st.sidebar.header(
    "Customer Information"
)


age = st.sidebar.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=35
)

tenure = st.sidebar.number_input(
    "Tenure (months)",
    min_value=1,
    max_value=100,
    value=5
)

monthly_charges = st.sidebar.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=90.0
)

login_frequency = st.sidebar.number_input(
    "Login Frequency",
    min_value=0,
    value=3
)

support_tickets = st.sidebar.number_input(
    "Support Tickets",
    min_value=0,
    value=6
)

payment_failures = st.sidebar.number_input(
    "Payment Failures",
    min_value=0,
    value=2
)

usage_hours = st.sidebar.number_input(
    "Usage Hours",
    min_value=0.0,
    value=10.0
)

contract = st.sidebar.selectbox(
    "Contract Type",
    [
        "Monthly",
        "Yearly",
        "Two Year"
    ]
)

subscription = st.sidebar.selectbox(
    "Subscription",
    [
        "Basic",
        "Standard",
        "Premium"
    ]
)


if st.button(
    "🔮 Predict Churn"
):

    customer = {

        "age": age,

        "tenure_months": tenure,

        "monthly_charges":
            monthly_charges,

        "login_frequency":
            login_frequency,

        "support_tickets":
            support_tickets,

        "payment_failures":
            payment_failures,

        "usage_hours":
            usage_hours,

        "contract_type":
            contract,

        "subscription_type":
            subscription
    }


    response = requests.post(
        "http://127.0.0.1:8000/predict",
        json=customer
    )


    if response.status_code == 200:

        result = response.json()


        st.metric(
            "Churn Probability",
            f"{result['churn_probability'] * 100:.2f}%"
        )


        st.write(
            "Risk Level:",
            result["risk"]
        )


        if result["risk"] == "CRITICAL":

            st.error(
                "🚨 Critical churn risk"
            )

        elif result["risk"] == "HIGH":

            st.warning(
                "⚠️ High churn risk"
            )

        else:

            st.success(
                "Customer is relatively safe"
            )