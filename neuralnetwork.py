from preprocessing import load_data
from preprocessing import monthly_sales

import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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
neural_network_model.add(Dense(32, activation='relu'))
neural_network_model.add(Dense(32, activation='relu'))

#Output layer (1 neuron for regression)
neural_network_model.add(Dense(1))

#Compile model
neural_network_model.compile(optimizer='adam', loss='mse')
