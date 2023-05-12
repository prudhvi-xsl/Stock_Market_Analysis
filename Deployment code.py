# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 15:38:22 2023

@author: golla
"""

import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import yfinance as yf


st.set_page_config(page_title="Reliance Stock Price Forecasting App")

st.title("Reliance Stock Price Forecasting App")

st.write("""
Enter a date range to see the predicted closing price for Reliance Industries Limited (RELIANCE.NS) for the next 30 days.
""")

start_date = st.date_input("Start date", value=pd.to_datetime('2015-01-01'))
end_date = st.date_input("End date", value=pd.to_datetime('2023-03-09'))

if start_date > end_date:
    st.error('Error: End date must be after start date.')

else:
    # Download stock data
    # Set the start and end date
    start_date = '2015-01-01'
    end_date = '2023-03-09'

    # Set the ticker
    ticker = 'RELIANCE.NS'

    # Get the data
    df = yf.download(ticker, start_date, end_date)

    # Create a new column with Date as index
    df['Date'] = df.index
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    df = df[(df['Date'] >= str(start_date)) & (df['Date'] <= str(end_date))]
    df = df.dropna()
    
    # Perform EDA
    st.write("## Exploratory Data Analysis")
    st.write("### Data Summary")
    st.write(df.describe())

    st.write("### Data Visualization")# Define function to load and preprocess data
    st.line_chart(df.Close)

@st.cache
def load_data():
    # Load data from yfinance
    df = yf.download("RELIANCE.NS", start="2015-01-01", end="2022-12-30")
    # Reset index to move date from upper header to main column
    df = df.reset_index()
    # Filter columns to only include date and Close price
    df = df[['Date', 'Close']]
    # Rename columns
    df.columns = ['date', 'Close']
    # Add features for year, month, day of week, and day of month
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_month'] = df['date'].dt.day
    # Convert date to string format
    df['date'] = df['date'].dt.strftime('%Y-%m-%d')
    return df

# Load data
df = load_data()

# Split data into training and testing sets
train = df[df['date'] < '2022-06-01']
test = df[df['date'] >= '2023-03-12']

# Define X and y for training and testing sets
X_train = train.drop(['date', 'Close'], axis=1)
y_train = train['Close']
X_test = test.drop(['date', 'Close'], axis=1)
y_test = test['Close']

# Train random forest regressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Make predictions for next 30 days starting from 2023-01-01
future_dates = pd.date_range(start='2023-03-15', periods=30, freq='D')
future_dates_str = [d.strftime('%Y-%m-%d') for d in future_dates]
future_features = pd.DataFrame({'date': future_dates_str})
future_features['year'] = future_features['date'].str.slice(0, 4).astype(int)
future_features['month'] = future_features['date'].str.slice(5, 7).astype(int)
future_features['day_of_week'] = pd.to_datetime(future_features['date']).dt.dayofweek
future_features['day_of_month'] = pd.to_datetime(future_features['date']).dt.day
future_features = future_features.drop('date', axis=1)
future_predictions = rf.predict(future_features)

# Create table of future dates and predicted Close prices
future_df = pd.DataFrame({'Date': future_dates, 'Predicted Close': future_predictions})
future_df['Date'] = future_df['Date'].dt.strftime('%Y-%m-%d')
future_df = future_df.set_index('Date')
st.table(future_df)


import matplotlib.pyplot as plt

# plot the line graph
plt.plot(future_predictions)

# add title and axis labels
plt.title("Predictions for Next 30 Days")
plt.xlabel("Days")
plt.ylabel("predictions")

# rotate x-axis labels for better readability
plt.xticks(rotation=45)
st.set_option('deprecation.showPyplotGlobalUse', False)
# display the plot
st.pyplot()

    
    
