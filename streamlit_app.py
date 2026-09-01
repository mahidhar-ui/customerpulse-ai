import streamlit as st
import requests


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="CustomerPulse AI",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# APPLICATION TITLE
# ============================================================

st.title(
    "🚀 CustomerPulse AI"
)

st.subheader(
    "AI-Powered Customer Churn Prediction"
)

st.write(
    "Enter customer information below to predict churn risk "
    "using the CustomerPulse AI prediction service."
)


# ============================================================
# BACKEND CONFIGURATION
# ============================================================

API_URL = "http://127.0.0.1:8000/predict"


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "👤 Customer Information"
)


age = st.sidebar.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=35,
    step=1
)


tenure = st.sidebar.number_input(
    "Tenure (months)",
    min_value=1,
    max_value=120,
    value=5,
    step=1
)


monthly_charges = st.sidebar.number_input(
    "Monthly Charges",
    min_value=0.0,
    max_value=10000.0,
    value=90.0,
    step=0.5
)


login_frequency = st.sidebar.number_input(
    "Login Frequency",
    min_value=0,
    max_value=1000,
    value=3,
    step=1
)


support_tickets = st.sidebar.number_input(
    "Support Tickets",
    min_value=0,
    max_value=1000,
    value=6,
    step=1
)


payment_failures = st.sidebar.number_input(
    "Payment Failures",
    min_value=0,
    max_value=100,
    value=2,
    step=1
)


usage_hours = st.sidebar.number_input(
    "Usage Hours",
    min_value=0.0,
    max_value=10000.0,
    value=10.0,
    step=0.5
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
    "Subscription Type",
    [
        "Basic",
        "Standard",
        "Premium"
    ]
)


# ============================================================
# CUSTOMER DATA
# ============================================================

customer = {

    "age": age,

    "tenure_months": tenure,

    "monthly_charges": monthly_charges,

    "login_frequency": login_frequency,

    "support_tickets": support_tickets,

    "payment_failures": payment_failures,

    "usage_hours": usage_hours,

    "contract_type": contract,

    "subscription_type": subscription
}


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_churn(customer_data):

    try:

        response = requests.post(
            API_URL,
            json=customer_data,
            timeout=30
        )

        return response

    except requests.exceptions.ConnectionError:

        return None

    except requests.exceptions.Timeout:

        return "TIMEOUT"

    except requests.exceptions.RequestException as e:

        return e


# ============================================================
# PREDICT CHURN BUTTON
# ============================================================

if st.button(
    "🔮 Predict Churn",
    use_container_width=True
):

    # --------------------------------------------------------
    # Show loading state
    # --------------------------------------------------------

    with st.spinner(
        "Analyzing customer data..."
    ):

        response = predict_churn(
            customer
        )


    # ========================================================
    # CONNECTION ERROR
    # ========================================================

    if response is None:

        st.error(
            "❌ Could not connect to the prediction server."
        )

        st.info(
            "Make sure your FastAPI backend is running at "
            "http://127.0.0.1:8000"
        )

        st.code(
            "uvicorn api:app --reload",
            language="bash"
        )


    # ========================================================
    # TIMEOUT
    # ========================================================

    elif response == "TIMEOUT":

        st.error(
            "⏱️ The prediction server took too long to respond."
        )


    # ========================================================
    # OTHER REQUEST ERROR
    # ========================================================

    elif not isinstance(response, requests.Response):

        st.error(
            f"❌ Request failed: {response}"
        )


    # ========================================================
    # SUCCESS
    # ========================================================

    elif response.status_code == 200:

        try:

            result = response.json()

        except ValueError:

            st.error(
                "❌ The server returned an invalid response."
            )

            st.code(
                response.text
            )

            st.stop()


        # ====================================================
        # READ RESULT
        # ====================================================

        churn_probability = float(
            result.get(
                "churn_probability",
                0
            )
        )

        risk = str(
            result.get(
                "risk",
                "UNKNOWN"
            )
        ).upper()


        prediction = result.get(
            "prediction",
            None
        )


        # ====================================================
        # RESULT SECTION
        # ====================================================

        st.divider()

        st.subheader(
            "📊 Churn Prediction Result"
        )


        # ====================================================
        # METRICS
        # ====================================================

        col1, col2, col3 = st.columns(
            3
        )


        with col1:

            st.metric(
                "Churn Probability",
                f"{churn_probability * 100:.2f}%"
            )


        with col2:

            if prediction is not None:

                if str(prediction) in ["1", "True", "true"]:

                    prediction_text = (
                        "Likely to Churn"
                    )

                else:

                    prediction_text = (
                        "Likely to Stay"
                    )

            else:

                if churn_probability >= 0.50:

                    prediction_text = (
                        "Likely to Churn"
                    )

                else:

                    prediction_text = (
                        "Likely to Stay"
                    )


            st.metric(
                "Prediction",
                prediction_text
            )


        with col3:

            st.metric(
                "Risk Level",
                risk
            )


        # ====================================================
        # RISK MESSAGE
        # ====================================================

        if risk == "CRITICAL":

            st.error(
                "🚨 Critical churn risk. "
                "Immediate retention action is recommended."
            )

        elif risk == "HIGH":

            st.warning(
                "⚠️ High churn risk. "
                "Consider proactive customer retention."
            )

        elif risk == "MEDIUM":

            st.warning(
                "⚠️ Medium churn risk. "
                "Monitor customer engagement."
            )

        elif risk == "LOW":

            st.success(
                "✅ Customer has relatively low churn risk."
            )

        else:

            st.info(
                f"Risk classification: {risk}"
            )


        # ====================================================
        # PROBABILITY PROGRESS BAR
        # ====================================================

        st.subheader(
            "📈 Churn Risk Indicator"
        )


        progress_value = max(
            0.0,
            min(
                1.0,
                churn_probability
            )
        )


        st.progress(
            progress_value
        )


        # ====================================================
        # BACKEND RECOMMENDATIONS
        # ====================================================

        recommendations = result.get(
            "recommendations",
            []
        )


        if recommendations:

            st.subheader(
                "💡 Retention Recommendations"
            )

            for recommendation in recommendations:

                st.write(
                    f"• {recommendation}"
                )


        # ====================================================
        # CUSTOMER SUMMARY
        # ====================================================

        st.subheader(
            "👤 Customer Summary"
        )


        summary_col1, summary_col2 = st.columns(
            2
        )


        with summary_col1:

            st.write(
                "**Age:**",
                age
            )

            st.write(
                "**Tenure:**",
                f"{tenure} months"
            )

            st.write(
                "**Monthly Charges:**",
                f"${monthly_charges:.2f}"
            )

            st.write(
                "**Login Frequency:**",
                login_frequency
            )

            st.write(
                "**Usage Hours:**",
                usage_hours
            )


        with summary_col2:

            st.write(
                "**Support Tickets:**",
                support_tickets
            )

            st.write(
                "**Payment Failures:**",
                payment_failures
            )

            st.write(
                "**Contract Type:**",
                contract
            )

            st.write(
                "**Subscription Type:**",
                subscription
            )


        # ====================================================
        # ENGINEERED FEATURES FROM BACKEND
        # ====================================================
        #
        # If your FastAPI backend returns these values, display
        # them here. The Streamlit frontend does NOT calculate
        # them because they should be generated using the same
        # formulas used during model training.
        # ====================================================

        engineered_features = result.get(
            "engineered_features",
            None
        )


        if engineered_features:

            st.subheader(
                "🔧 Model Features"
            )

            feature_col1, feature_col2, feature_col3, feature_col4 = (
                st.columns(4)
            )


            with feature_col1:

                st.metric(
                    "Customer Value",
                    f"{engineered_features.get('customer_value', 0):.2f}"
                )


            with feature_col2:

                st.metric(
                    "Usage / Login",
                    f"{engineered_features.get('usage_per_login', 0):.2f}"
                )


            with feature_col3:

                st.metric(
                    "Payment Risk",
                    f"{engineered_features.get('payment_risk', 0):.2f}"
                )


            with feature_col4:

                st.metric(
                    "Support Intensity",
                    f"{engineered_features.get('support_intensity', 0):.2f}"
                )


        # ====================================================
        # RAW SERVER RESPONSE
        # ====================================================

        with st.expander(
            "🔧 Technical Prediction Details"
        ):

            st.json(
                result
            )


    # ========================================================
    # SERVER ERROR
    # ========================================================

    else:

        st.error(
            f"❌ Prediction server returned "
            f"HTTP {response.status_code}"
        )


        # Try to show backend error message

        try:

            error_data = response.json()

            st.json(
                error_data
            )

        except ValueError:

            st.code(
                response.text
            )


# ============================================================
# API STATUS
# ============================================================

st.divider()

st.subheader(
    "🔌 Prediction Service"
)


if st.button(
    "Check API Status"
):

    try:

        health_response = requests.get(
            "http://127.0.0.1:8000/",
            timeout=5
        )


        if health_response.status_code == 200:

            st.success(
                "✅ Prediction API is running."
            )

            st.write(
                health_response.text
            )

        else:

            st.warning(
                f"API responded with "
                f"status code {health_response.status_code}"
            )


    except requests.exceptions.ConnectionError:

        st.error(
            "❌ Prediction API is not running."
        )

        st.info(
            "Start your FastAPI server before making predictions."
        )

    except requests.exceptions.Timeout:

        st.error(
            "⏱️ API health check timed out."
        )

    except Exception as e:

        st.error(
            f"API check failed: {e}"
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "CustomerPulse AI • "
    "Churn Prediction API"
)
