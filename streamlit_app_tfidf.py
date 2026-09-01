import os
import re
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from pathlib import Path
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="CustomerPulse AI",
    page_icon="🚀",
    layout="wide"
)


# ============================================================
# APPLICATION TITLE
# ============================================================

st.title("🚀 CustomerPulse AI")

st.subheader("AI-Powered Customer Churn Prediction")

st.write(
    "Predict customer churn risk and retrieve relevant customer "
    "policies using TF-IDF based document search."
)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DOCUMENTS_DIR = BASE_DIR / "documents"

MODEL_FILES = [
    BASE_DIR / "model.pkl",
    BASE_DIR / "churn_model.pkl",
    BASE_DIR / "customer_churn_model.pkl",
    BASE_DIR / "models" / "model.pkl",
    BASE_DIR / "models" / "churn_model.pkl"
]


# ============================================================
# LOAD MACHINE LEARNING MODEL
# ============================================================

@st.cache_resource
def load_model():

    for model_path in MODEL_FILES:

        if model_path.exists():

            try:

                model = joblib.load(model_path)

                return model, str(model_path)

            except Exception as e:

                st.error(
                    f"Unable to load model: {e}"
                )

                return None, None

    return None, None


model, model_path = load_model()


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

def extract_pdf_text(pdf_path):

    text = ""

    try:

        reader = PdfReader(str(pdf_path))

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:

                text += page_text + "\n"

    except Exception as e:

        st.warning(
            f"Could not read {pdf_path.name}: {e}"
        )

    return text


# ============================================================
# SPLIT DOCUMENT INTO CHUNKS
# ============================================================

def create_chunks(text, chunk_size=800, overlap=100):

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    if not text:

        return []

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        if chunk.strip():

            chunks.append(
                chunk.strip()
            )

        start += chunk_size - overlap

    return chunks


# ============================================================
# LOAD ALL DOCUMENTS
# ============================================================

@st.cache_data
def load_documents():

    all_chunks = []

    if not DOCUMENTS_DIR.exists():

        return all_chunks

    pdf_files = list(
        DOCUMENTS_DIR.glob("*.pdf")
    )

    for pdf_file in pdf_files:

        text = extract_pdf_text(
            pdf_file
        )

        chunks = create_chunks(
            text
        )

        for chunk in chunks:

            all_chunks.append(
                {
                    "source": pdf_file.name,
                    "text": chunk
                }
            )

    return all_chunks


documents = load_documents()


# ============================================================
# CREATE TF-IDF VECTOR DATABASE
# ============================================================

@st.cache_resource
def create_tfidf_index(documents):

    if not documents:

        return None, None, None

    texts = [
        document["text"]
        for document in documents
    ]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=10000
    )

    matrix = vectorizer.fit_transform(
        texts
    )

    return vectorizer, matrix, texts


vectorizer, tfidf_matrix, document_texts = create_tfidf_index(
    documents
)


# ============================================================
# TF-IDF SEARCH FUNCTION
# ============================================================

def search_documents(query, top_k=3):

    if (
        vectorizer is None
        or tfidf_matrix is None
        or not query.strip()
    ):

        return []

    query_vector = vectorizer.transform(
        [query]
    )

    similarity_scores = cosine_similarity(
        query_vector,
        tfidf_matrix
    ).flatten()

    top_indices = np.argsort(
        similarity_scores
    )[::-1][:top_k]

    results = []

    for index in top_indices:

        score = similarity_scores[index]

        if score <= 0:

            continue

        results.append(
            {
                "source": documents[index]["source"],
                "text": documents[index]["text"],
                "score": float(score)
            }
        )

    return results


# ============================================================
# RISK CLASSIFICATION
# ============================================================

def get_risk(probability):

    if probability < 0.30:

        return "LOW"

    elif probability < 0.60:

        return "MEDIUM"

    else:

        return "HIGH"


# ============================================================
# RETENTION RECOMMENDATION
# ============================================================

def get_recommendation(
    probability,
    payment_failures,
    support_tickets,
    login_frequency,
    contract_type
):

    recommendations = []

    if probability >= 0.60:

        recommendations.append(
            "Prioritize this customer for immediate retention outreach."
        )

    elif probability >= 0.30:

        recommendations.append(
            "Monitor the customer and consider a proactive engagement."
        )

    else:

        recommendations.append(
            "Customer appears relatively stable. Continue normal engagement."
        )

    if payment_failures > 0:

        recommendations.append(
            "Investigate payment failures and offer payment assistance."
        )

    if support_tickets >= 5:

        recommendations.append(
            "Review unresolved support issues and improve customer support."
        )

    if login_frequency <= 3:

        recommendations.append(
            "Consider a re-engagement campaign because of low login frequency."
        )

    if contract_type == "Monthly":

        recommendations.append(
            "Consider offering a longer-term plan or loyalty incentive."
        )

    return recommendations


# ============================================================
# FEATURE ENGINEERING
# ============================================================
#
# IMPORTANT:
# These four features are required by your trained model:
#
#   payment_risk
#   usage_per_login
#   support_intensity
#   customer_value
#
# The formulas below are reasonable defaults.
#
# Ideally, use the EXACT SAME formulas used when training
# your model.
# ============================================================

def create_engineered_features(df):

    df = df.copy()

    # --------------------------------------------------------
    # 1. USAGE PER LOGIN
    # --------------------------------------------------------
    #
    # Measures how many hours of usage occur per login.
    #
    # Example:
    # usage_hours = 20
    # login_frequency = 10
    #
    # usage_per_login = 2.0
    # --------------------------------------------------------

    df["usage_per_login"] = (
        df["usage_hours"]
        / df["login_frequency"].replace(0, 1)
    )


    # --------------------------------------------------------
    # 2. PAYMENT RISK
    # --------------------------------------------------------
    #
    # Payment failures are used as a simple risk indicator.
    #
    # More payment failures -> higher payment risk.
    # --------------------------------------------------------

    df["payment_risk"] = (
        df["payment_failures"]
    )


    # --------------------------------------------------------
    # 3. SUPPORT INTENSITY
    # --------------------------------------------------------
    #
    # Measures support tickets relative to tenure.
    #
    # More support tickets over a shorter tenure means
    # higher support intensity.
    # --------------------------------------------------------

    df["support_intensity"] = (
        df["support_tickets"]
        / df["tenure_months"].replace(0, 1)
    )


    # --------------------------------------------------------
    # 4. CUSTOMER VALUE
    # --------------------------------------------------------
    #
    # Simple customer value estimate:
    #
    # Monthly charges × tenure
    #
    # Example:
    # 75 × 12 = 900
    # --------------------------------------------------------

    df["customer_value"] = (
        df["monthly_charges"]
        * df["tenure_months"]
    )


    return df


# ============================================================
# MODEL FEATURE INSPECTION
# ============================================================

def get_required_model_columns(model):

    """
    Attempts to identify the columns expected by the
    trained sklearn pipeline.
    """

    required_columns = None

    try:

        # ----------------------------------------------------
        # ColumnTransformer
        # ----------------------------------------------------

        if hasattr(
            model,
            "named_steps"
        ):

            for step_name, step in model.named_steps.items():

                if hasattr(
                    step,
                    "transformers_"
                ):

                    transformer_columns = []

                    for (
                        name,
                        transformer,
                        columns
                    ) in step.transformers_:

                        if columns is not None:

                            if isinstance(
                                columns,
                                (list, tuple)
                            ):

                                transformer_columns.extend(
                                    columns
                                )

                    if transformer_columns:

                        required_columns = list(
                            dict.fromkeys(
                                transformer_columns
                            )
                        )

                        return required_columns


        # ----------------------------------------------------
        # Direct ColumnTransformer
        # ----------------------------------------------------

        if hasattr(
            model,
            "transformers_"
        ):

            transformer_columns = []

            for (
                name,
                transformer,
                columns
            ) in model.transformers_:

                if columns is not None:

                    if isinstance(
                        columns,
                        (list, tuple)
                    ):

                        transformer_columns.extend(
                            columns
                        )

            if transformer_columns:

                required_columns = list(
                    dict.fromkeys(
                        transformer_columns
                    )
                )

    except Exception:

        pass

    return required_columns


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "Customer Information"
)


age = st.sidebar.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=35,
    step=1
)


tenure_months = st.sidebar.number_input(
    "Tenure (months)",
    min_value=0,
    max_value=120,
    value=12,
    step=1
)


monthly_charges = st.sidebar.number_input(
    "Monthly Charges",
    min_value=0.0,
    max_value=1000.0,
    value=75.5,
    step=0.5
)


login_frequency = st.sidebar.number_input(
    "Login Frequency",
    min_value=0,
    max_value=100,
    value=20,
    step=1
)


support_tickets = st.sidebar.number_input(
    "Support Tickets",
    min_value=0,
    max_value=100,
    value=2,
    step=1
)


payment_failures = st.sidebar.number_input(
    "Payment Failures",
    min_value=0,
    max_value=100,
    value=0,
    step=1
)


usage_hours = st.sidebar.number_input(
    "Usage Hours",
    min_value=0.0,
    max_value=1000.0,
    value=15.5,
    step=0.5
)


contract_type = st.sidebar.selectbox(
    "Contract Type",
    [
        "Monthly",
        "Yearly",
        "Two Year"
    ]
)


subscription_type = st.sidebar.selectbox(
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

    "tenure_months": tenure_months,

    "monthly_charges": monthly_charges,

    "login_frequency": login_frequency,

    "support_tickets": support_tickets,

    "payment_failures": payment_failures,

    "usage_hours": usage_hours,

    "contract_type": contract_type,

    "subscription_type": subscription_type
}


# ============================================================
# MAIN PREDICTION
# ============================================================

if st.button(
    "🔮 Predict Churn",
    use_container_width=False
):

    if model is None:

        st.error(
            "ML model not found. "
            "Please place your trained model as model.pkl "
            "or churn_model.pkl in the project directory."
        )

    else:

        try:

            # ------------------------------------------------
            # CREATE DATAFRAME
            # ------------------------------------------------

            input_df = pd.DataFrame(
                [customer]
            )


            # ------------------------------------------------
            # CREATE ENGINEERED FEATURES
            # ------------------------------------------------

            input_df = create_engineered_features(
                input_df
            )


            # ------------------------------------------------
            # DISPLAY GENERATED FEATURES
            # ------------------------------------------------

            with st.expander(
                "🔧 Engineered Features"
            ):

                engineered_display = pd.DataFrame(
                    {
                        "Feature": [
                            "customer_value",
                            "usage_per_login",
                            "payment_risk",
                            "support_intensity"
                        ],

                        "Value": [
                            input_df["customer_value"].iloc[0],
                            input_df["usage_per_login"].iloc[0],
                            input_df["payment_risk"].iloc[0],
                            input_df["support_intensity"].iloc[0]
                        ]
                    }
                )

                st.dataframe(
                    engineered_display,
                    use_container_width=True,
                    hide_index=True
                )


            # ------------------------------------------------
            # MODEL COLUMN CHECK
            # ------------------------------------------------

            required_columns = get_required_model_columns(
                model
            )

            if required_columns:

                missing_columns = [
                    column
                    for column in required_columns
                    if column not in input_df.columns
                ]

                if missing_columns:

                    st.error(
                        "The trained model requires columns "
                        "that are not present in the input data:"
                    )

                    st.code(
                        "\n".join(
                            missing_columns
                        )
                    )

                    st.info(
                        "This usually means the training and "
                        "prediction feature engineering do not match."
                    )

                    st.stop()


            # ------------------------------------------------
            # PREDICTION
            # ------------------------------------------------

            prediction = model.predict(
                input_df
            )[0]


            # ------------------------------------------------
            # PROBABILITY
            # ------------------------------------------------

            if hasattr(
                model,
                "predict_proba"
            ):

                probabilities = model.predict_proba(
                    input_df
                )[0]


                # --------------------------------------------
                # Determine probability of churn
                # --------------------------------------------

                if hasattr(
                    model,
                    "classes_"
                ):

                    classes = model.classes_

                    if 1 in classes:

                        churn_index = list(
                            classes
                        ).index(1)

                        churn_probability = float(
                            probabilities[churn_index]
                        )

                    else:

                        churn_probability = float(
                            probabilities[-1]
                        )

                else:

                    if len(probabilities) > 1:

                        churn_probability = float(
                            probabilities[1]
                        )

                    else:

                        churn_probability = float(
                            probabilities[0]
                        )

            else:

                churn_probability = float(
                    prediction
                )


            # ------------------------------------------------
            # PROTECT AGAINST INVALID PROBABILITY
            # ------------------------------------------------

            churn_probability = max(
                0.0,
                min(
                    1.0,
                    churn_probability
                )
            )


            # ------------------------------------------------
            # RISK
            # ------------------------------------------------

            risk = get_risk(
                churn_probability
            )


            # =================================================
            # DISPLAY RESULTS
            # =================================================

            st.divider()

            st.subheader(
                "📊 Churn Prediction Result"
            )


            col1, col2, col3 = st.columns(
                3
            )


            with col1:

                st.metric(
                    "Churn Probability",
                    f"{churn_probability * 100:.2f}%"
                )


            with col2:

                st.metric(
                    "Prediction",
                    "Likely to Churn"
                    if prediction == 1
                    else "Likely to Stay"
                )


            with col3:

                st.metric(
                    "Risk Level",
                    risk
                )


            # =================================================
            # RISK MESSAGE
            # =================================================

            if risk == "HIGH":

                st.error(
                    "⚠️ Customer has a high churn risk."
                )

            elif risk == "MEDIUM":

                st.warning(
                    "⚠️ Customer has a medium churn risk."
                )

            else:

                st.success(
                    "✅ Customer is relatively safe."
                )


            # =================================================
            # RETENTION RECOMMENDATIONS
            # =================================================

            st.subheader(
                "💡 Retention Recommendations"
            )


            recommendations = get_recommendation(
                churn_probability,
                payment_failures,
                support_tickets,
                login_frequency,
                contract_type
            )


            for recommendation in recommendations:

                st.write(
                    f"• {recommendation}"
                )


            # =================================================
            # CUSTOMER SUMMARY
            # =================================================

            st.subheader(
                "👤 Customer Summary"
            )


            summary_df = pd.DataFrame(
                {
                    "Feature": [

                        "Age",

                        "Tenure",

                        "Monthly Charges",

                        "Login Frequency",

                        "Support Tickets",

                        "Payment Failures",

                        "Usage Hours",

                        "Contract Type",

                        "Subscription Type",

                        "Customer Value",

                        "Usage Per Login",

                        "Payment Risk",

                        "Support Intensity"
                    ],

                    "Value": [

                        age,

                        tenure_months,

                        monthly_charges,

                        login_frequency,

                        support_tickets,

                        payment_failures,

                        usage_hours,

                        contract_type,

                        subscription_type,

                        round(
                            float(
                                input_df[
                                    "customer_value"
                                ].iloc[0]
                            ),
                            2
                        ),

                        round(
                            float(
                                input_df[
                                    "usage_per_login"
                                ].iloc[0]
                            ),
                            2
                        ),

                        round(
                            float(
                                input_df[
                                    "payment_risk"
                                ].iloc[0]
                            ),
                            2
                        ),

                        round(
                            float(
                                input_df[
                                    "support_intensity"
                                ].iloc[0]
                            ),
                            2
                        )
                    ]
                }
            )


            st.dataframe(
                summary_df,
                use_container_width=True,
                hide_index=True
            )


        except Exception as e:

            st.error(
                "Prediction failed."
            )

            st.exception(e)


# ============================================================
# DOCUMENT SEARCH / TF-IDF RAG
# ============================================================

st.divider()

st.header(
    "📚 Customer Policy & Knowledge Search"
)

st.write(
    "Ask a question about customer support, retention, "
    "subscriptions, or product features."
)


query = st.text_input(
    "Enter your question",
    placeholder=(
        "Example: What can we do when a customer wants to cancel?"
    )
)


top_k = st.slider(
    "Number of results",
    min_value=1,
    max_value=5,
    value=3
)


if st.button(
    "🔎 Search Policies"
):

    if not query.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        results = search_documents(
            query,
            top_k
        )


        if not results:

            st.info(
                "No relevant information was found "
                "in the uploaded documents."
            )

        else:

            st.subheader(
                "Relevant Information"
            )


            for i, result in enumerate(
                results,
                start=1
            ):

                st.markdown(
                    f"### Result {i}"
                )

                st.caption(
                    f"Source: {result['source']} | "
                    f"TF-IDF Similarity: "
                    f"{result['score']:.3f}"
                )

                st.write(
                    result["text"]
                )

                st.divider()


# ============================================================
# SYSTEM INFORMATION
# ============================================================

with st.expander(
    "🔧 System Information"
):

    st.write(
        "### Machine Learning Model"
    )


    if model_path:

        st.success(
            f"Loaded: {model_path}"
        )

    else:

        st.warning(
            "No ML model found."
        )


    st.write(
        "### TF-IDF Knowledge Base"
    )


    st.write(
        f"Documents loaded: {len(documents)}"
    )


    if vectorizer is not None:

        st.write(
            f"Vocabulary size: "
            f"{len(vectorizer.vocabulary_)}"
        )

        st.write(
            f"Chunks indexed: "
            f"{len(documents)}"
        )

    else:

        st.warning(
            "No PDF documents were indexed."
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "CustomerPulse AI • Churn Prediction + TF-IDF Knowledge Retrieval"
)
