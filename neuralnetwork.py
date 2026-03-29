from preprocessing import load_data

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor

# load and preprocess dataset
X_train, X_test, y_train, y_test = load_data()


# Train Model
MLPRegressor = MLPRegressor(hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=200, random_state=42)
MLPRegressor.fit(X_train, y_train)
# evaluate model
y_pred = MLPRegressor.predict(X_test)
MAE = mean_absolute_error(y_test, y_pred)
RMSE = mean_squared_error(y_test, y_pred) ** 0.5
R2 = r2_score(y_test, y_pred)
print(f"MAE: {MAE}")
print(f"RMSE: {RMSE}")
print(f"R2: {R2}")


