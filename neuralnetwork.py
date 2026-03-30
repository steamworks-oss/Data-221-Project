from preprocessing import load_data

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# load and preprocess dataset
X_train, X_test, y_train, y_test = load_data()

#initiate model
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)  # output layer for regression
])

model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)


#Train Model

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)


#Evaluate Model
y_pred = model.predict(X_test).flatten()

MAE = mean_absolute_error(y_test, y_pred)
RMSE = mean_squared_error(y_test, y_pred) ** 0.5
R2 = r2_score(y_test, y_pred)

print(f"MAE: {MAE}")
print(f"RMSE: {RMSE}")
print(f"R2: {R2}")
