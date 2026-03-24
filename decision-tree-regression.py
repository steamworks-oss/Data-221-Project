import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

# load dataset
df = pd.read_csv('Sales.csv')

# ------------------
# Preprocessing plan
# ------------------

# Convert date strings to Timestamp object, and then sort DateFrame by date in ascending order
df["Date"] = pd.to_datetime(df["Date"], format='%Y-%m-%d')
df.sort_values(by=["Date"], inplace=True)

# Create new DateFrame grouped by total unit sales in a month
# df_monthly = df.groupby([df["Date"].dt.year, df["Date"].dt.month])["Order_Quantity"].sum()

# Initialize features matrix X and labels vector y
X = df.drop(columns=["Order_Quantity"], axis=1)
y = df["Order_Quantity"]

# One-hot encode categorical columns

# Split chronologically with a roughly 70% train, 30% test split
split = int(len(df) * 0.7)
X_train = X[:split]
X_test = X[split:]
y_train = y[:split]
y_test = y[split:]

# Standardize the input features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)