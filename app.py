import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import time

# Set page title and layout
st.set_page_config(page_title="Stock Price Predictor", page_icon="📈", layout="wide")
#st.set_page_config(page_title="Live Stock Price Tracker", page_icon="📈", layout="wide")

# Title
st.title("📊 Stock Price Prediction Using LSTM")
st.markdown("### Predict stock prices with deep learning models.")

# Sidebar - User Input
st.sidebar.markdown("📌 **For Indian stocks, use:** `RELIANCE.NS`, `TCS.NS`, etc.")
st.sidebar.header("🔍 Select Stock & Date Range")
stock_symbol = st.sidebar.text_input("Enter Stock Symbol (e.g., AAPL, TSLA):", "AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2023-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-01-01"))

# Load pre-trained model
model = load_model("lstm_stock_model.h5")
scaler = joblib.load("scaler.pkl")

# Fetch stock data
st.subheader(f"📌 Latest Stock Data for {stock_symbol}")
try:
    stock_data = yf.Ticker(stock_symbol).history(start=start_date, end=end_date)
    st.dataframe(stock_data.tail(10))  # Show last 10 records

    # Extract closing prices
    closing_prices = stock_data['Close'].values.reshape(-1, 1)
    closing_prices_scaled = scaler.transform(closing_prices)

    # Prepare data for prediction (last 60 days)
    X_input = closing_prices_scaled[-60:].reshape(1, 60, 1)
    
    # Prediction
    with st.spinner("🔄 Predicting..."):
        predicted_stock_price = model.predict(X_input)
        predicted_stock_price = scaler.inverse_transform(predicted_stock_price)
    st.success("✅ Prediction Complete!")

    # Display Prediction
    st.subheader("📈 Predicted Stock Price for Next Day")
    st.write(f"**Predicted Price: ${predicted_stock_price[0][0]:.2f}**")

    # Plot Interactive Graph
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name="Actual Price"))
    fig.add_trace(go.Scatter(x=[stock_data.index[-1] + pd.Timedelta(days=1)], y=[predicted_stock_price[0][0]], 
                             mode='markers', name="Predicted Price", marker=dict(color="red", size=10)))
    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error("⚠️ Error fetching data. Please check the stock symbol and try again.")

# Footer
st.markdown("---")
st.markdown("🔗 **Developed by [Surjeet Singh]** | Powered by LSTM & Streamlit")


# Sidebar - User Input for Stock Symbol
st.sidebar.header("🔍 Enter Stock Symbol")
stock_symbol = st.sidebar.text_input("Stock Symbol (e.g., AAPL, RELIANCE.NS)", "RELIANCE.NS")

# Live Stock Price Display
st.title(f"📈 Live Stock Price for {stock_symbol}")

# Create a placeholder for live price
live_price = st.empty()

# Fetch and Update Price Every 5 Seconds
while True:
    try:
        stock_data = yf.Ticker(stock_symbol)
        current_price = stock_data.history(period="1d")['Close'].iloc[-1]  # Get latest closing price
        live_price.metric(label="💰 Current Price", value=f"${current_price:.2f}")

    except Exception as e:
        live_price.error("⚠️ Error fetching live stock price. Please check the stock symbol.")
    
    time.sleep(5)  # Refresh every 5 seconds

