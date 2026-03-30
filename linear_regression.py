import pandas as pd

from preprocessing import load_data
from preprocessing import monthly_sales

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Loads preprocessed dataset.
X_train, X_test, y_train, y_test = load_data()

# Trains model.
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluates model.
y_pred = model.predict(X_test)

# Computes mean absolute error, mean squared error and r squared.
MAE = mean_absolute_error(y_test, y_pred)
RMSE = mean_squared_error(y_test, y_pred) ** 0.5
R2 = r2_score(y_test, y_pred)

# Prints computations to user.
print('MAE:', MAE)
print('RMSE:', RMSE)
print('R2:', R2)

# Gets unique months from dataset.
months = monthly_sales["Date"].drop_duplicates().sort_values()
# Train-test split (70/30).
split = int(len(months) * 0.7)
# Selects months used for testing.
test_months = months.iloc[:split]
# Checks if Date from each row in monthly_sales is in test_months.
months_test = monthly_sales["Date"].isin(test_months)

# Creates DataFrame with prepared data.
results = pd.DataFrame({
    "Date": monthly_sales.loc[months_test, "Date"],
    "Actual": y_test.values,
    "Predicted": y_pred
})

# Groups data by month-year and sum values for each month.
plot_values = results.groupby("Date")[["Actual", "Predicted"]].sum().reset_index()