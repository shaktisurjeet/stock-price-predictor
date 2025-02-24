import numpy as np
import pandas as pd
import yfinance as yf
import joblib
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Fetch historical stock data
stock = "AAPL"  # Change to your desired stock
df = yf.download(stock, period="5y")

# Extract 'Close' prices
data = df[['Close']].values

# Normalize data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data)

# Save the scaler
joblib.dump(scaler, 'scaler.pkl')
print("✅ Scaler saved as scaler.pkl")

# Create sequences for LSTM
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

seq_length = 100  # Use last 100 days
X, y = create_sequences(scaled_data, seq_length)

# Split data into training and testing sets (80% train, 20% test)
split = int(len(X) * 0.8)
X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

# Build LSTM model
model = Sequential([
    LSTM(100, return_sequences=True, input_shape=(seq_length, 1)),
    Dropout(0.2),
    LSTM(100, return_sequences=False),
    Dropout(0.2),
    Dense(50, activation='relu'),
    Dense(1)
])

# Compile model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Save model
model.save('lstm_stock_model.h5')
print("✅ Model saved as lstm_stock_model.h5")
