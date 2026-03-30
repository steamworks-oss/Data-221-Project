import pandas as pd
import numpy as np

def load_data():
    # load the csv file
    df = pd.read_csv("Sales.csv")
    
    # only keep bike sales
    df = df[df["Product_Category"] == "Bikes"]

    # turn the Date column into actual dates
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")

    # sort everything by date so it's in order
    df = df.sort_values("Date")

    # group by month + state and add up the quantities
    monthly_sales = df.groupby([df["Date"].dt.to_period("M"), "State"])["Order_Quantity"].sum().reset_index()

    # rename the columns so they look nicer
    monthly_sales.columns = ["Date", "State", "Order_Quantity"]

    # convert the period back to a normal timestamp
    monthly_sales["Date"] = monthly_sales["Date"].dt.to_timestamp()

    # make some basic time features
    monthly_sales["Year"] = monthly_sales["Date"].dt.year
    monthly_sales["Month"] = monthly_sales["Date"].dt.month
    monthly_sales["Quarter"] = monthly_sales["Date"].dt.quarter

    # cyclical encoding for months (so Dec and Jan aren't far apart)
    monthly_sales["Month_sin"] = np.sin(2 * np.pi * monthly_sales["Month"] / 12)
    monthly_sales["Month_cos"] = np.cos(2 * np.pi * monthly_sales["Month"] / 12)

    # sort by state + date so lag features work properly
    monthly_sales = monthly_sales.sort_values(["State", "Date"])

    # lag features (previous months)
    monthly_sales["lag_1"] = monthly_sales.groupby("State")["Order_Quantity"].shift(1)
    monthly_sales["lag_2"] = monthly_sales.groupby("State")["Order_Quantity"].shift(2)
    monthly_sales["lag_3"] = monthly_sales.groupby("State")["Order_Quantity"].shift(3)

    # rolling averages (shifted so we don't cheat)
    monthly_sales["rolling_mean_3"] = monthly_sales.groupby("State")["Order_Quantity"].transform(
        lambda x: x.shift(1).rolling(3).mean()
    )
    monthly_sales["rolling_mean_6"] = monthly_sales.groupby("State")["Order_Quantity"].transform(
        lambda x: x.shift(1).rolling(6).mean()
    )

    # drop rows that don't have enough history for lags
    monthly_sales = monthly_sales.dropna().reset_index(drop=True)

    # pick the features we want to use
    feature_cols = [
        "Year", "Month", "Quarter",
        "Month_sin", "Month_cos",
        "lag_1", "lag_2", "lag_3",
        "rolling_mean_3", "rolling_mean_6",
        "State"
    ]

    # X = inputs, y = target
    X = monthly_sales[feature_cols]
    y = monthly_sales["Order_Quantity"]

    # turn State into one-hot columns
    X = pd.get_dummies(X)

    # split by time (70% train, 30% test)
    months = monthly_sales["Date"].drop_duplicates().sort_values()
    split = int(len(months) * 0.7)

    train_months = months.iloc[:split]
    test_months = months.iloc[split:]

    train_mask = monthly_sales["Date"].isin(train_months)
    test_mask = monthly_sales["Date"].isin(test_months)

    X_train = X[train_mask]
    X_test = X[test_mask]
    y_train = y[train_mask]
    y_test = y[test_mask]

    return X_train, X_test, y_train, y_test, monthly_sales, train_mask, test_mask
