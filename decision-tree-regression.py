import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

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

# Add features to the new DataFrame to use as predictors
monthly_sales["Year"] = monthly_sales["Date"].dt.year
monthly_sales["Month"] = monthly_sales["Date"].dt.month

# Initialize features matrix X and labels vector y
X = monthly_sales[["Year", "Month", "State"]]
y = monthly_sales["Order_Quantity"]

# One-hot encode categorical columns
X = pd.get_dummies(X)

# Split chronologically with a roughly 70% train, 30% test split
split = int(len(monthly_sales) * 0.7)
X_train = X.iloc[:split]
X_test = X.iloc[split:]
y_train = y.iloc[:split]
y_test = y.iloc[split:]

# Standardize the input features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
