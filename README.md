🚀 CustomerPulse AI

AI-powered customer churn prediction and customer intelligence
application.

CustomerPulse AI combines Machine Learning, FastAPI, Streamlit, SQL
analytics, and Retrieval-Augmented Generation (RAG) to identify
customers who may churn and provide useful insights from internal
customer-support and business documents.

✨ Features

🤖 Customer churn prediction using a trained ML model

📊 Churn probability and LOW/MEDIUM/HIGH risk classification

⚡ FastAPI REST backend

📖 Swagger/OpenAPI documentation

🎨 Interactive Streamlit dashboard

📚 PDF document ingestion for RAG

🔎 Recursive text splitting

🧠 OpenAI embeddings

🗃️ Chroma vector database

🗄️ SQL-based customer analytics

💡 Explainability/business insight module

🐳 Docker support for deployment

🏗️ Architecture

Customer Data
     |
     v
Feature Engineering
     |
     v
ML Training --> Trained Model
                    |
                    v
              FastAPI /predict
                    |
                    v
             Streamlit Dashboard


PDF Documents
     |
     v
   ingest.py
     |
     v
Text Splitting
     |
     v
OpenAI Embeddings
     |
     v
Chroma Vector DB
     |
     v
RAG Retrieval

📂 Project Structure

CustomerPulse-AI/
│
├── data/
├── documents/
│   ├── customer_support_policy.pdf
│   ├── product_features.pdf
│   ├── retention_strategy.pdf
│   └── subscription_policy.pdf
├── models/
├── notebooks/
├── rag/
│   └── vector_db/
├── venv/
├── analytics.sql
├── database.py
├── explain.py
├── features.py
├── generate_data.py
├── ingest.py
├── load_database.py
├── main.py
├── predict.py
├── train.py
├── requirements.txt
├── .env
├── .gitignore
└── README.md

The Streamlit frontend should be present as streamlit_app.py if you
are using the dashboard.

🛠️ Technology Stack

Area                Technology

Language            Python
Machine Learning    Scikit-learn
Backend             FastAPI
API Server          Uvicorn
Frontend            Streamlit
Database            SQL
Vector Database     Chroma
Embeddings          OpenAI Embeddings
RAG                 LangChain components
Data Processing     Pandas / NumPy
Containerization    Docker
API Documentation   Swagger / OpenAPI

⚙️ Installation

1. Clone the repository

git clone <YOUR_GITHUB_REPOSITORY_URL>
cd CustomerPulse-AI

2. Create and activate a virtual environment

Windows PowerShell:

python -m venv venv
.\venv\Scripts\Activate.ps1

3. Install dependencies

pip install -r requirements.txt

If required by the RAG pipeline:

pip install langchain-community langchain-text-splitters langchain-openai chromadb pypdf

🔐 Environment Variables

Create a .env file in the project root:

OPENAI_API_KEY=your_openai_api_key

Never commit the real API key to GitHub.

Recommended .gitignore entries:

.env
venv/
__pycache__/
*.pyc
rag/vector_db/

🧠 Machine Learning Pipeline

Generate data

python generate_data.py

Train the model

python train.py

The trained model is stored under models/ according to the
implementation in train.py.

Test prediction

python predict.py

Example result:

{
    "churn_probability": 0.5398,
    "prediction": 1,
    "risk": "MEDIUM"
}

⚡ FastAPI Backend

Start the backend:

python -m uvicorn main:app --reload

API:

http://127.0.0.1:8000

Swagger documentation:

http://127.0.0.1:8000/docs

API Endpoints

GET /

Checks that the API is running.

Example:

{
  "message": "CustomerPulse AI API is running"
}

POST /predict

Example request:

{
  "age": 35,
  "tenure_months": 12,
  "monthly_charges": 75.5,
  "login_frequency": 20,
  "support_tickets": 2,
  "payment_failures": 0,
  "usage_hours": 15.5,
  "contract_type": "Monthly",
  "subscription_type": "Standard"
}

Example response:

{
  "churn_probability": 0.6321,
  "prediction": 1,
  "risk": "HIGH"
}

Risk classification is determined by the thresholds implemented in the
application.

🎨 Streamlit Dashboard

Start FastAPI first:

python -m uvicorn main:app --reload

Keep that terminal running.

Open a second terminal:

cd C:\Users\<YOUR_USERNAME>\CustomerPulse-AI
.\venv\Scripts\Activate.ps1

Then run:

streamlit run streamlit_app.py

The dashboard normally opens at:

http://localhost:8501

Important

If Streamlit calls:

http://127.0.0.1:8000/predict

the FastAPI server must be running before clicking Predict Churn.

📚 RAG Pipeline

The current knowledge base contains:

documents/
├── customer_support_policy.pdf
├── product_features.pdf
├── retention_strategy.pdf
└── subscription_policy.pdf

Run ingestion:

python ingest.py

The pipeline is:

PDF files
   ↓
PDF Loader
   ↓
Documents
   ↓
RecursiveCharacterTextSplitter
   ↓
Text Chunks
   ↓
OpenAI Embeddings
   ↓
Chroma
   ↓
rag/vector_db/

A successful run reports the number of documents loaded and chunks
created.

Example:

Loaded 24 documents.
Created 31 chunks.

PDF validation

Input files must be real PDF files. A valid PDF normally begins with:

%PDF

If ingestion reports invalid pdf header, replace the invalid file with
a valid PDF.

🗄️ Database and SQL Analytics

Database functionality is handled through:

database.py
load_database.py
analytics.sql

Load data:

python load_database.py

SQL analytics are maintained in:

analytics.sql

Possible analysis areas include:

Customer churn

Subscription behavior

Support tickets

Payment failures

Customer retention

Customer segmentation

💡 Explainability

explain.py contains the project's explanation/business-insight
functionality.

The goal is to make model predictions understandable to business users
instead of displaying only a probability score.

🐳 Docker

Check Docker:

docker --version

Build:

docker build -t customerpulse-ai .

Run:

docker run -p 8000:8000 customerpulse-ai

Docker Desktop on Windows requires hardware virtualization and the
required Windows virtualization components.

🧪 API Testing

Use Swagger:

http://127.0.0.1:8000/docs

Open POST /predict.

Click Try it out.

Enter customer information.

Click Execute.

Check the returned churn probability and risk.

You can also test from Python:

import requests

customer = {
    "age": 35,
    "tenure_months": 12,
    "monthly_charges": 75.5,
    "login_frequency": 20,
    "support_tickets": 2,
    "payment_failures": 0,
    "usage_hours": 15.5,
    "contract_type": "Monthly",
    "subscription_type": "Standard"
}

response = requests.post(
    "http://127.0.0.1:8000/predict",
    json=customer
)

print(response.json())

🚨 Troubleshooting

ModuleNotFoundError: No module named 'api'

Make sure the Uvicorn command matches the actual FastAPI file:

python -m uvicorn main:app --reload

Here:

main means main.py

app means the FastAPI application object

Connection refused from Streamlit

Start the backend:

python -m uvicorn main:app --reload

Then start Streamlit in another terminal:

streamlit run streamlit_app.py

Missing credentials from OpenAI

Add the API key to .env:

OPENAI_API_KEY=your_key_here

Then restart the terminal/application.

PDF ingestion error

If you see invalid pdf header, verify that the files in documents/
are genuine PDFs and not text/HTML files renamed with .pdf.

Docker virtualization error

If Docker Desktop reports that virtualization support is not detected:

Enable CPU virtualization in BIOS/UEFI.

Enable the required Windows virtualization features.

Restart Windows.

Start Docker Desktop again.

Run:

docker --version

🔒 Security

Never commit:

.env

API keys

Passwords

Database credentials

Private customer information

Use environment variables for secrets.

🎯 End-to-End Workflow

                CUSTOMER CHURN SYSTEM

Customer Data
     ↓
Feature Engineering
     ↓
ML Model
     ↓
Churn Probability
     ↓
Risk Classification
     ↓
FastAPI
     ↓
Streamlit Dashboard


                RAG SYSTEM

Internal PDFs
     ↓
Document Ingestion
     ↓
Text Chunking
     ↓
Embeddings
     ↓
Chroma Vector Database
     ↓
Semantic Retrieval
     ↓
Customer Support / Retention Insights

🚀 Future Improvements

RAG-powered customer-support chatbot

Retention recommendation engine

Customer segmentation

SHAP-based explainability

Prediction history

Authentication and authorization

Production database

Model monitoring

Automated tests

Docker Compose

CI/CD pipeline

Cloud deployment

Role-based dashboards

Automated retention recommendations

👨‍💻 Author

G. V. Mahidhar Reddy

CustomerPulse AI --- AI-powered customer churn prediction and customer
intelligence project.

📄 License

This project is intended for educational, portfolio, and demonstration
purposes.

Add an appropriate open-source license if you plan to distribute the
project publicly.
