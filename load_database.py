import pandas as pd

from database import engine


df = pd.read_csv(
    "data/customers.csv"
)


df.to_sql(
    "customers",
    engine,
    if_exists="replace",
    index=False
)

print(
    "Customer data loaded into PostgreSQL."
)