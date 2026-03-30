from preprocessing import load_data
from preprocessing import monthly_sales

import matplotlib.pyplot as plt
import pandas as pd

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
tf.random.set_seed(42) #seed for reproducable results

#load and preprocess dataset
X_train, X_test, y_train, y_test = load_data()

#Convert to float32 for TensorFlow (Chatgpt helped with this code to fix an error I was getting about data types)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')

#Build Neural Network Model
neural_network_model = Sequential()

#Input layer
neural_network_model.add(InputLayer(input_shape=(X_train.shape[1],)))

#Hidden layers
neural_network_model.add(Dense(64, activation='relu')) #density of 64
neural_network_model.add(Dense(32, activation='relu')) #density of 32

#Output layer (1 neuron for regression)
neural_network_model.add(Dense(1))

#Compile model
neural_network_model.compile(optimizer='adam', loss='mse')

#train model
history = neural_network_model.fit(X_train, y_train, epochs=50, validation_split=0.2)

#Evaluate
y_pred = neural_network_model.predict(X_test).flatten()

MAE = mean_absolute_error(y_test, y_pred)
RMSE = mean_squared_error(y_test, y_pred) ** 0.5
R2 = r2_score(y_test, y_pred)

print(f"MAE: {MAE}")
print(f"RMSE: {RMSE}")
print(f"R2: {R2}")

# get
df = pd.read_csv('Sales.csv')

#Put test dates, actual values, and predicted values together
months = monthly_sales["Date"].drop_duplicates().sort_values()
split = int(len(months) * 0.7)
test_months = months.iloc[split:]
months_test = monthly_sales["Date"].isin(test_months)

results = pd.DataFrame({
    "Date": monthly_sales.loc[months_test, "Date"],
    "Actual": y_test.values,
    "Predicted": y_pred
})

# Group by month-year and sum quantities for each date
plot_values = results.groupby("Date")[["Actual", "Predicted"]].sum().reset_index()

# Plot
plt.plot(plot_values["Date"], plot_values["Actual"], label="Actual", color="blue")
plt.plot(plot_values["Date"], plot_values["Predicted"], label="Predicted", color="red")
plt.xlabel("Date")
plt.ylabel("Quantity Ordered")
plt.title("Actual vs Predicted Quantity Ordered by Month-Year")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig("actual_vs_predicted_neural_network.png")
