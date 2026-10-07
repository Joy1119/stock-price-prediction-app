import pandas as pd
import numpy as np
from keras.models import load_model
import tensorflow as tf
import streamlit as st

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score, classification_report
from keras.losses import BinaryCrossentropy
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
# Extracting stock data from yahoo
import yfinance as yf
from pandas_datareader import data as pdr
# yf.pdr_override()
import datetime

# start = '2010-01-01'
# end = '2019-12-31'

today = datetime.datetime.now()
next_year = today.year - 1
jan_1 = datetime.datetime(next_year, 1, 1)
dec_31 = datetime.datetime(next_year, 12, 31)
max_val = datetime.datetime(next_year + 1, 12, 31)

start = st.date_input('Start Date', jan_1, max_value=max_val)
end = st.date_input('End Date', dec_31, max_value=max_val)

if start is not None and end is not None:
    if start >= end:
        st.error('End date must be greater than the start date')
        st.stop()

company_list = {
    "GOOGLE" : "GOOG",
    "MICROSOFT" : "MSFT",
    "AMAZON" : "AMZN"
}

st.title('Stock Price Prediction') 
option = st.selectbox(
    'Select a company to predict',
    ('GOOGLE', 'MICROSOFT', 'AMAZON'))

selected = company_list[option]

if selected and start and end:

    # df = pdr.get_data_yahoo(selected, start, end)
    df = yf.download(selected, start, end)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    st.subheader('Stock Data')
    st.dataframe(df)


    st.subheader('Closing Price vs Time Chart')
    fig = plt.figure(figsize = (12,6))
    plt.plot(df.Close)
    st.pyplot(fig)

    st.subheader('Closing Price vs Time Chat with 100MA')
    ma100 = df.Close.rolling(100).mean()
    fig = plt.figure(figsize = (12,6))
    plt.plot(ma100)
    plt.plot(df.Close)
    st.pyplot(fig)

    st.subheader('Closing Price vs Time Chat with 100MA & 200MA')
    ma100 = df.Close.rolling(100).mean()
    ma200 = df.Close.rolling(200).mean()
    fig = plt.figure(figsize = (12,6))
    plt.plot(ma100, 'r')
    plt.plot(ma200, 'g')
    plt.plot(df.Close, 'b')
    st.pyplot(fig)



    # Splitting Data into Training and Testing

############################## Amazon #################################

    AMZ_data = pd.DataFrame(df.Close)
    AMZ_dataset = AMZ_data.values


    AMZ_training_data_len = int(np.ceil( len(AMZ_dataset) * .80 ))
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0,1))

    AMZ_scaled_data = scaler.fit_transform(AMZ_dataset)

    AMZ_test_data = AMZ_scaled_data[AMZ_training_data_len - 100: , :]

    L_s_test = []

    for i in range(100, len(AMZ_test_data)):

        L_s_test.append(AMZ_test_data[i-100:i, 0])

    L_s_test= np.array(L_s_test)

    L_t_test = AMZ_dataset[AMZ_training_data_len:, :]
############################# Google #################################################
    GG_data = pd.DataFrame(df.Close)
    GG_dataset = GG_data.values


    GG_training_data_len = int(np.ceil( len(GG_dataset) * .80 ))
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0,1))

    GG_scaled_data = scaler.fit_transform(GG_dataset)

    GG_test_data = GG_scaled_data[GG_training_data_len - 100: , :]
    L_m_test = []

    for i in range(100, len(GG_test_data)):

        L_m_test.append(GG_test_data[i-100:i, 0])

    L_m_test= np.array(L_m_test)


    # model Loading
    L_n_test = GG_dataset[GG_training_data_len:, :]
    
  ########################## Microsoft ##############################################  
    MS_data = pd.DataFrame(df.Close)
    MS_dataset = MS_data.values


    MS_training_data_len = int(np.ceil( len(MS_dataset) * .80 ))
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0,1))

    MS_scaled_data = scaler.fit_transform(MS_dataset)

    MS_test_data = MS_scaled_data[MS_training_data_len - 100: , :]
    L_a_test = []

    for i in range(100, len(MS_test_data)):

        L_a_test.append(MS_test_data[i-100:i, 0])

    L_a_test= np.array(L_a_test)


    # model Loading
    L_b_test = MS_dataset[MS_training_data_len:, :]
 #########################################################################################  

    def plot_predictions_vs_original(model_file, test_data, original_data, title):
        # Load the model
        model = load_model(model_file)

        # Get predictions
        predictions = model.predict(test_data)

        # Inverse transform predictions
        predictions = scaler.inverse_transform(predictions)

        # Plot predictions vs original
        fig = plt.figure(figsize=(12, 6))
        plt.plot(predictions, 'r', label='Predicted Price')
        plt.plot(original_data, 'g', label='Original Price')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.title(title)
        plt.show()
        st.pyplot(fig)

# Example usage
    
    import os

current_dir = os.getcwd()



AMZ_model_file = os.path.join(current_dir, 'AMZ_Stock_Market_Prediction.h5')
GG_model_file = os.path.join(current_dir, 'GG_Stock_Market_Prediction.h5')
MS_model_file = os.path.join(current_dir, 'MS_Stock_Market_Prediction.h5')
print(GG_model_file)
if not os.path.exists(GG_model_file):
    raise FileNotFoundError(f"Model file not found: {GG_model_file}")    
if selected == 'AMZN':
    plot_predictions_vs_original(AMZ_model_file, L_s_test, L_t_test, 'Amazon Predictions vs Original')
elif selected == 'MSFT':
    plot_predictions_vs_original(MS_model_file, L_a_test, L_b_test, 'Microsoft Predictions vs Original')
elif selected == 'GOOG':
    plot_predictions_vs_original(GG_model_file, L_m_test, L_n_test, 'Google Predictions vs Original')
    
else:
    st.write('Select a company, start and end date')



   