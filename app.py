import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression

# Load data
df = pd.read_csv('final.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by='Date')

# Feature engineering
df['Open_prev'] = df['Open'].shift(1)
df['Volume_prev'] = df['Volume'].shift(1)
df['RSI_prev'] = df['RSI'].shift(1)
df['MACD_prev'] = df['MACD'].shift(1)
df['Volatility_10_prev'] = df['Volatility_10'].shift(1)
df = df.dropna()

# Model
X = df[['Open_prev','Volume_prev','RSI_prev','MACD_prev','Volatility_10_prev']]
y = df['Close']

model = LinearRegression()
model.fit(X, y)

# ---------------- UI ----------------
st.set_page_config(layout="wide")
st.title("📊 Stock Analysis Dashboard")

# Sidebar filters
start_date = st.sidebar.date_input("Start Date", df['Date'].min())
end_date = st.sidebar.date_input("End Date", df['Date'].max())

filtered_df = df[(df['Date'] >= str(start_date)) & (df['Date'] <= str(end_date))]

# -------- Objective 1 --------
st.subheader("📈 Price Trend")
fig1 = px.line(filtered_df, x='Date', y='Close', title="Stock Price Over Time")
st.plotly_chart(fig1, use_container_width=True)

# -------- Objective 2 --------
st.subheader("📊 Volume vs Return")
fig2 = px.scatter(filtered_df, x='Volume', y='Return', title="Volume vs Return")
st.plotly_chart(fig2, use_container_width=True)

# -------- Objective 3 --------
st.subheader("📉 Volatility Comparison")
fig3 = px.line(filtered_df, x='Date', y=['Volatility_10','Volatility_30'])
st.plotly_chart(fig3, use_container_width=True)

# -------- Objective 4 --------
st.subheader("📍 Indicators vs Return")
indicator = st.selectbox("Choose Indicator", ['RSI','MACD'])
fig4 = px.scatter(filtered_df, x=indicator, y='Return')
st.plotly_chart(fig4, use_container_width=True)

# -------- Prediction --------
st.subheader("🤖 Predict Next Close Price")

open_val = st.number_input("Open", value=float(df['Open'].iloc[-1]))
volume_val = st.number_input("Volume", value=float(df['Volume'].iloc[-1]))
rsi_val = st.number_input("RSI", value=float(df['RSI'].iloc[-1]))
macd_val = st.number_input("MACD", value=float(df['MACD'].iloc[-1]))
vol_val = st.number_input("Volatility_10", value=float(df['Volatility_10'].iloc[-1]))

if st.button("Predict"):
    pred = model.predict([[open_val, volume_val, rsi_val, macd_val, vol_val]])
    st.success(f"Predicted Close Price: {pred[0]:.2f}")