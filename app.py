import streamlit as st
import numpy as np
import yfinance as yf
import joblib
import tensorflow as tf

# Load saved model and scaler
model = tf.keras.models.load_model('lstm_stock_model.h5')
scaler = joblib.load('scaler.pkl')

# Streamlit UI
st.title("📈 Stock Price Prediction with LSTM")
st.markdown("Enter a stock ticker to predict the next day's closing price.")

# User input for stock ticker
stock = st.text_input("Enter Stock Ticker (e.g., AAPL)", "AAPL")

if st.button("Predict"):
    try:
        # Fetch latest stock data
        data = yf.download(stock, period="100d")['Close'].values.reshape(-1,1)
        
        # Normalize using saved scaler
        data_scaled = scaler.transform(data)

        # Prepare input for LSTM (last 100 days)
        input_data = np.array([data_scaled[-100:]])  # Use last 100 days
        prediction = model.predict(input_data)

        # Convert back to original price scale
        predicted_price = scaler.inverse_transform(prediction)[0][0]
        
        # Display result
        st.success(f"📊 Predicted Closing Price for {stock}: **${predicted_price:.2f}**")

    except Exception as e:
        st.error("⚠️ Error fetching stock data. Please check the ticker symbol and try again.")
