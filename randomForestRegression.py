import pandas as pd
from preprocessing import load_data
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt


# Load dataset
X_train, X_test, y_train, y_test, monthly_sales, train_dates, test_dates = load_data()

# Train Model
random_forest_regressor = RandomForestRegressor(
    n_estimators=100,   #
    max_depth=12,
    min_samples_leaf=3,
    min_samples_split=8,
    random_state=42
)

random_forest_regressor.fit(X_train, y_train)

importance_feature = random_forest_regressor.feature_importances_
print("importance feature: ", importance_feature[:10])
# Evaluate Model

y_pred = random_forest_regressor.predict(X_test)

MAE = mean_absolute_error(y_test, y_pred)
RMSE = mean_squared_error(y_test, y_pred) ** 0.5
R2 = r2_score(y_test, y_pred)

print(f"MAE: {MAE}")
print(f"RMSE: {RMSE}")
print(f"R2: {R2}")


# Display Results

results = pd.DataFrame({
    "Date": monthly_sales.loc[test_dates, "Date"],
    "Actual": y_test.values,
    "Predicted": y_pred
})

# Group by month-year and sum
plot_values = results.groupby("Date")[["Actual", "Predicted"]].sum().reset_index()

# Plot
plt.figure("Random Forest Regression")

plt.plot(plot_values["Date"], plot_values["Actual"], label="Actual", color="blue")
plt.plot(plot_values["Date"], plot_values["Predicted"], label="Predicted", color="red")

plt.xlabel("Date")
plt.ylabel("Quantity Ordered")
plt.title("Actual vs Predicted Quantity Ordered by Month-Year")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()
