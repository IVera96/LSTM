# ========== IMPORTS ==========
import pandas as pd
import numpy as np
import json
from matplotlib import pyplot as plt

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# ========== LOAD & PREPARE DATA =========

# Import data
df = pd.read_csv('data.csv')

# Transform month into cyclical features
df['month_sin'] = np.sin(df['Month'] * np.pi / 6)
df['month_cos'] = np.cos(df['Month'] * np.pi / 6)

# Select relevant features
df = df[['Year', 'Month', 'value', 'month_sin', 'month_cos']]

# Combine 'Month' and 'Year' into a single datetime column
df["date"] = df["Month"].map(str) + df["Year"].map(str)
df["date"] = df["date"].astype(str).str.zfill(6)
df["date"] = pd.to_datetime(df["date"], format="%m%Y")

# Drop unnecessary columns and sort
df.drop(columns=["Month", "Year"], inplace=True)
df.sort_values(by="date", inplace=True)
df.reset_index(drop=True, inplace=True)

# Drop the last 7 rows (e.g., NaN targets or reserved for test)
df = df[:-7]

# ==========  TRAIN/TEST SPLIT ==========

train_split = -12
most_recent_date = df["date"].values[train_split]
data = df[['value', 'month_sin', 'month_cos']].values

# ==========  SEQUENCE CREATION FUNCTION ==========

def create_lstm_sequences(df, seq_length):
    """
    Convert multivariate time series into LSTM sequences.
    Output shape: (n_samples, seq_length, n_features)
    """
    X = []
    for i in range(len(df) - seq_length + 1):
        X.append(df[i:i + seq_length])
    return np.array(X)

# ==========  NORMALIZATION ==========

scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(data)

# Create sequences of length 12
X = create_lstm_sequences(scaled, 12)

# Train/test data split
n_train_hours = len(X) - 12
train = X[:n_train_hours, :]
test = scaled[-12:, :]

# Prepare LSTM inputs
train_X = train[:-1, :, :]
train_y = scaled[12:n_train_hours + 11, 0]

# The test input is the last sequence
test_X = train[-1, :, :].reshape((1, 12, 3))
test_y = test

# Print input shapes
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# ==========  BUILD & TRAIN LSTM MODEL ==========

model = Sequential()
model.add(LSTM(200, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')

# Train model
history = model.fit(train_X, train_y, epochs=200, batch_size=16, verbose=0, shuffle=False)

# ==========  FORECASTING FUNCTION ==========

def forecast_recursive(model, last_sequence, n_steps, scaler, feature_index=0):
    """
    Recursive multi-step forecasting using LSTM.
    Each prediction is used as input for the next.
    """
    predictions = []
    current_seq = last_sequence.copy()

    for i in range(n_steps):
        # Predict next value
        next_pred_scaled = model.predict(current_seq, verbose=0)

        # Create complete feature array for inverse_transform
        complete_array = np.asarray([next_pred_scaled[0, 0], test[i, 1], test[i, 2]]).reshape(1, -1)
        next_pred_unscaled = scaler.inverse_transform(complete_array)

        # Store prediction
        predictions.append(next_pred_unscaled.flatten()[0])

        # Update current sequence for next iteration
        current_seq = np.concatenate(
            [current_seq[:, 1:, :], complete_array.reshape(1, 1, 3)],
            axis=1
        )

    return np.array(predictions)

# ==========  RUN FORECASTING ==========

n_steps = 12
last_sequence = test_X
predictions_recursive = forecast_recursive(model, last_sequence, n_steps, scaler, feature_index=0)

# ==========  EVALUATION METRICS ==========

mse_recursive = mean_squared_error(data[-12:, 0], predictions_recursive)
mae_recursive = mean_absolute_error(data[-12:, 0], predictions_recursive)
mape_recursive = mean_absolute_percentage_error(data[-12:, 0], predictions_recursive)
r2_recursive = 1 - (mse_recursive / np.var(data[-12:, 0]))
rmse_recursive = np.sqrt(mse_recursive)

print(f"Recursive LSTM Mean Squared Error (MSE): {mse_recursive}")
print(f"Recursive LSTM Mean Absolute Error (MAE): {mae_recursive}")
print(f"Recursive LSTM Mean Absolute Percentage Error (MAPE): {mape_recursive}")
print(f"Recursive LSTM Root Mean Squared Error (RMSE): {rmse_recursive}")
print(f"Recursive LSTM R^2: {r2_recursive}")

# ========== PLOTTING RESULTS ==========

plt.figure(figsize=(12, 6))
plt.plot(df["date"], df["value"], label="Train Data", color="blue")
plt.plot(
    df["date"],
    list(df["value"][:train_split]) + list(predictions_recursive),
    label="Recursive LSTM Prediction",
    color="purple"
)
plt.axvline(x=most_recent_date, color='red', linestyle='--', label='Most Recent Date')
plt.xlabel("Date")
plt.ylabel("Value")
plt.title('Recursive LSTM Predictions vs Actual Data')
plt.legend()
plt.show()
