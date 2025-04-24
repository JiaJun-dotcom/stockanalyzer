import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Fetch Historical Stock Data, specify ticker
ticker = 'TSLA'  

# Define the date range
start_date = '2020-01-01'
end_date = '2025-04-23'

# Download stock data from Yahoo Finance
data = yf.download(ticker, start=start_date, end=end_date)

# Log first few rows of data
print(data.head())

# Preprocessing the Data (MinMax Scaling)
data = data[['Close']]  # Use only the 'Close' price for prediction

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Prepare the data for the model
def prepare_data(data, time_step=60):
    X, y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i - time_step:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

time_step = 60  # Look-back period (60 days)
X, y = prepare_data(scaled_data, time_step)

# Reshape X for LSTM input [samples, time steps, features]
X = X.reshape(X.shape[0], X.shape[1], 1)

# Train-validation split:
split = int(len(X) * 0.8) # Split into 80-20 train-validation data.
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

# Build the LSTM Model
model = Sequential()

# Add LSTM layers
model.add(LSTM(units=128, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=64, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=32, return_sequences=False))
model.add(Dropout(0.2))

# Output layer
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Add Early Stopping to prevent overfitting(stops if loss doesnt improve over 10 epochs)
early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

# Train the model(used 50 epochs for training)
model.fit(
    X_train, y_train, 
    validation_data = (X_val, y_val),
    epochs=50, 
    batch_size=32, 
    callbacks=[early_stopping],
    verbose=1
)

# Make Predictions

# Get the last 60 days of stock data for prediction
test_data = data[-time_step:].values
test_data = scaler.transform(test_data)
X_test = []
X_test.append(test_data)
X_test = np.array(X_test)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Predict the stock price for the next day
predicted_price = model.predict(X_test)
predicted_price = scaler.inverse_transform(predicted_price)

print(f"Predicted next day's stock price for {ticker}: ${predicted_price[0][0]} USD")

# Plot the Results

# Predict on the entire dataset for visualization
full_data = data[['Close']].values
scaled_full_data = scaler.fit_transform(full_data)

X_full, y_full = prepare_data(scaled_full_data, time_step)
X_full = X_full.reshape(X_full.shape[0], X_full.shape[1], 1)
predicted_prices = model.predict(X_full)
predicted_prices = scaler.inverse_transform(predicted_prices)

# Plot the actual vs predicted prices
plt.figure(figsize=(14, 6))
plt.plot(data.index, data['Close'], label='Actual Price', color='red')
plt.plot(data.index[time_step:], predicted_prices, label='Predicted Price', color='green')
plt.title(f'{ticker} Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()