import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

# load dataset
df = pd.read_csv('Sales.csv')

# print(df["State"].nunique())

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

# Initialize features matrix X and labels vector y
X = monthly_sales[["Year", "Month", "State"]]
y = monthly_sales["Order_Quantity"]

# One-hot encode categorical columns
X = pd.get_dummies(X)

# Split chronologically with a roughly 70% train, 30% test split
split = int(len(monthly_sales) * 0.7) # divide by the number of states to get a clean divide between months to prevent data leakage
X_train = X.iloc[:split]
X_test = X.iloc[split:]
y_train = y.iloc[:split]
y_test = y.iloc[split:]

# ------------------
# Train Model
# ------------------
decision_tree_regressor = DecisionTreeRegressor(max_depth=10, min_samples_leaf=4, min_samples_split=10, random_state=42)
decision_tree_regressor.fit(X_train, y_train)

# evaluate model
y_pred = decision_tree_regressor.predict(X_test)

MAE = mean_absolute_error(y_test, y_pred)
RMSE = mean_squared_error(y_test, y_pred) ** 0.5
R2 = r2_score(y_test, y_pred)

print(f"MAE: {MAE}")
print(f"RMSE: {RMSE}")
print(f"R2: {R2}")

# ------------------
# Display results
# ------------------

# Plot the actual quantities ordered by month in a line graph
actual_values = monthly_sales.groupby("Date")["Order_Quantity"].sum()

plt.plot(actual_values.index, actual_values.values)
plt.xlabel("Date")
plt.ylabel("Quantity Ordered")
plt.title("Quantity Ordered per Month")
plt.xticks(rotation=45)
plt.show()

# Plot the predicted quantities ordered by month in a line graph