from sqlalchemy import create_engine
DATABASE_URL = "postgresql+psycopg2://postgres:postgress%40123@localhost:5433/customerpulse"

engine = create_engine(DATABASE_URL)

print("PostgreSQL connection configured successfully.")