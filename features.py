import pandas as pd


def create_features(df):

    df = df.copy()

    # Average monthly usage
    df["usage_per_login"] = (
        df["usage_hours"] /
        (df["login_frequency"] + 1)
    )

    # Support intensity
    df["support_intensity"] = (
        df["support_tickets"] /
        (df["tenure_months"] + 1)
    )

    # Payment risk
    df["payment_risk"] = (
        df["payment_failures"] /
        (df["tenure_months"] + 1)
    )

    # Customer value
    df["customer_value"] = (
        df["monthly_charges"] *
        df["tenure_months"]
    )

    return df