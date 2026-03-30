from preprocessing import load_data

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

