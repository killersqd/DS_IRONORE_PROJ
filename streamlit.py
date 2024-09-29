import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load dataset
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, index_col=0, parse_dates=True)
    st.write("DataFrame", df.head())
    
    columns_to_forecast = df.columns.tolist()
    
    # Dropdown to select the column to forecast
    column_to_forecast = st.selectbox("Select the column to forecast", columns_to_forecast)
    
    if column_to_forecast:
        # Load the pre-trained model
        model_filename = f'GBM_model_{column_to_forecast}.pkl'
        model = joblib.load(model_filename)
        
        # Split the data into train and test sets
        train = df[:-60]  # Assuming you want to keep the last 60 days for testing
        test = df[-60:]
        
        # Creating the features
        X_train = np.arange(len(train)).reshape(-1, 1)
        X_test = np.arange(len(train), len(train) + len(test)).reshape(-1, 1)
        
        # Predict button
        if st.button('Predict'):
            # Forecasting for the next two months
            future_index = np.arange(len(train), len(train) + 60).reshape(-1, 1)
            future_forecast = model.predict(future_index)
            
            # Create a DataFrame for the forecast
            forecast_dates = pd.date_range(start=train.index[-1] + pd.Timedelta(days=1), periods=60)
            forecast_df = pd.DataFrame(future_forecast, index=forecast_dates, columns=[column_to_forecast])
            
            # Plotting the forecast
            plt.figure(figsize=(10, 6))
            plt.plot(train.index, train[column_to_forecast], label='Train')
            plt.plot(test.index, test[column_to_forecast], label='Test')
            plt.plot(forecast_df.index, forecast_df[column_to_forecast], label='Forecast')
            plt.legend()
            plt.title(f'Forecast for {column_to_forecast}')
            plt.xlabel('Date')
            plt.ylabel(column_to_forecast)
            plt.grid(True)
            st.pyplot(plt)
            
            # Display forecast
            st.write("Forecasted Values", forecast_df.head(60))
