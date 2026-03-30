import pandas as pd
import matplotlib.pyplot as plt

from preprocessing import load_data

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression

# Loads preprocessed dataset.
X_train, X_test, y_train, y_test, monthly_sales = load_data()

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

# Creates DataFrame with prepared data.
results = pd.DataFrame({
    "Date": monthly_sales.loc[y_test.index, "Date"],
    "Actual": y_test.values,
    "Predicted": y_pred
})

# Groups data by month-year and sum values for each month.
plot_values = results.groupby("Date")[["Actual", "Predicted"]].sum().reset_index()

# Plots actual and predicted values over time.
plt.plot(plot_values["Date"], plot_values["Actual"], label="Actual", color="blue")
plt.plot(plot_values["Date"], plot_values["Predicted"], label="Predicted", color="red")
# Adds labels to axes.
plt.xlabel("Date")
plt.ylabel("Quantity Ordered")
# Adds title.
plt.title("Actual vs Predicted Quantity Ordered by Month-Year")
# Rotates x-axis label.
plt.xticks(rotation=45)
# Shows legend.
plt.legend()
# Adjusts layout to prevent overlapping.
plt.tight_layout()
# Display plot.
plt.show()