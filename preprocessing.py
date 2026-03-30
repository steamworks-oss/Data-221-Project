import pandas as pd
import numpy as np

def load_data():
    # load file
    df = pd.read_csv("Sales.csv")

    # only bikes
    df = df[df["Product_Category"] == "Bikes"]

    # fix dates
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    # monthly totals per state
    monthly = df.groupby([df["Date"].dt.to_period("M"), "State"])["Order_Quantity"].sum().reset_index()
    monthly["Date"] = monthly["Date"].dt.to_timestamp()

    # time features
    monthly["Year"] = monthly["Date"].dt.year
    monthly["Month"] = monthly["Date"].dt.month
    monthly["Quarter"] = monthly["Date"].dt.quarter

    # cyclical month encoding
    monthly["Month_sin"] = np.sin(2 * np.pi * monthly["Month"] / 12)
    monthly["Month_cos"] = np.cos(2 * np.pi * monthly["Month"] / 12)

    # sort for lag features
    monthly = monthly.sort_values(["State", "Date"])

    # lag features
    monthly["lag_1"] = monthly.groupby("State")["Order_Quantity"].shift(1)
    monthly["lag_2"] = monthly.groupby("State")["Order_Quantity"].shift(2)
    monthly["lag_3"] = monthly.groupby("State")["Order_Quantity"].shift(3)

    # rolling averages (shifted to avoid leakage)
    monthly["rolling_mean_3"] = monthly.groupby("State")["Order_Quantity"].shift(1).rolling(3).mean()
    monthly["rolling_mean_6"] = monthly.groupby("State")["Order_Quantity"].shift(1).rolling(6).mean()

    # drop rows missing lag values
    monthly = monthly.dropna().reset_index(drop=True)

    # features + target
    features = [
        "Year", "Month", "Quarter",
        "Month_sin", "Month_cos",
        "lag_1", "lag_2", "lag_3",
        "rolling_mean_3", "rolling_mean_6",
        "State"
    ]

    X = pd.get_dummies(monthly[features])
    y = monthly["Order_Quantity"]

    # time split
    months = monthly["Date"].drop_duplicates().sort_values()
    split = int(len(months) * 0.7)

    train_months = months[:split]
    test_months = months[split:]

    train_dates = monthly["Date"].isin(train_months)
    test_dates = monthly["Date"].isin(test_months)


    X_train = X[train_dates]
    X_test = X[test_dates]
    y_train = y[train_dates]
    y_test = y[test_dates]

    return X_train, X_test, y_train, y_test, monthly, train_dates, test_dates
