import pandas as pd

# load dataset
df = pd.read_csv('Sales.csv')

# ------------------
# Preprocessing plan
# ------------------

# Filter out non-bicycle sales
df = df[df["Product_Category"] == "Bikes"]

# Convert date strings to Timestamp object, and then sort DateFrame by date in ascending order
df["Date"] = pd.to_datetime(df["Date"], format='%Y-%m-%d')
df.sort_values(by=["Date"], inplace=True)

# Create new DataFrame grouped by total unit sales in a month in a location
monthly_sales = df.groupby([df["Date"].dt.to_period('M'), "State"])["Order_Quantity"].sum().reset_index()
monthly_sales.columns = ["Date", "State", "Order_Quantity"]
monthly_sales["Date"] = monthly_sales["Date"].dt.to_timestamp()

# Add features to the new DataFrame to use as predictors
monthly_sales["Year"] = monthly_sales["Date"].dt.year
monthly_sales["Month"] = monthly_sales["Date"].dt.month

# Function that returns the train/test data
def load_data():
    # Initialize features matrix X and labels vector y
    X = monthly_sales[["Year", "Month", "State"]]
    y = monthly_sales["Order_Quantity"]

    # One-hot encode categorical columns
    X = pd.get_dummies(X)

    # Split chronologically with a roughly 70% train, 30% test split
    # Splits by months instead of rows to avoid data leakage (code generated with the help of ChatGPT)
    months = monthly_sales["Date"].drop_duplicates().sort_values()
    split = int(len(months) * 0.7)
    train_months = months.iloc[:split]
    test_months = months.iloc[split:]

    months_train = monthly_sales["Date"].isin(train_months)
    months_test = monthly_sales["Date"].isin(test_months)

    X_train = X.loc[months_train]
    X_test = X.loc[months_test]
    y_train = y.loc[months_train]
    y_test = y.loc[months_test]

    return X_train, X_test, y_train, y_test
