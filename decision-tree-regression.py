import pandas as pd

from preprocessing import load_data

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor, plot_tree
import matplotlib.pyplot as plt

# load dataset
X_train, X_test, y_train, y_test, monthly_sales, train_dates, test_dates = load_data()

# ------------------
# Train Model
# ------------------
decision_tree_regressor = DecisionTreeRegressor(random_state=42) # max_depth=10 min_samples_leaf=4, min_samples_split=10
decision_tree_regressor.fit(X_train, y_train)

# Evaluate model
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

#Put test dates, actual values, and predicted values together
results = pd.DataFrame({
    "Date": monthly_sales.loc[test_dates, "Date"],
    "Actual": y_test.values,
    "Predicted": y_pred
})

# Group by month-year and sum quantities for each date
plot_values = results.groupby("Date")[["Actual", "Predicted"]].sum().reset_index()

# Plot the predicted and actual values in a line graph
plt.plot(plot_values["Date"], plot_values["Actual"], label="Actual", color="blue")
plt.plot(plot_values["Date"], plot_values["Predicted"], label="Predicted", color="red")
plt.xlabel("Date")
plt.ylabel("Quantity Ordered")
plt.title("Actual vs Predicted Quantity Ordered by Month-Year")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# Display Decision Tree
plot_tree(
    decision_tree_regressor,
    feature_names=X_train.columns,
    filled=True,
    rounded=True,
    fontsize=8,
    max_depth=3
)
plt.tight_layout()
plt.show()