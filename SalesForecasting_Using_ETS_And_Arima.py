#!/usr/bin/env python
# coding: utf-8

# Leveraging sales data from a Kaggle competition, I applied advanced time series forecasting techniques, including ARIMA and Exponential Smoothing (ETS), to predict future sales trends and enhance forecasting accuracy.

# In[15]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error


# In[2]:


#Reading the data from CSV file train.csv and 
data = pd.read_csv("train.csv")
data_transactions = pd.read_csv("transactions.csv")


# In[3]:


#Processing the data
data_transactions['date'] = pd.to_datetime(data_transactions['date'], format='%Y-%m-%d')
data_transactions.set_index('date', inplace=True)


# In[4]:


#Observing the data  (Date is kept as index)
data_transactions.head()


# In[10]:


#Selecting a specific store for prediction, lets say 54
store_data = data_transactions[data_transactions['store_nbr'] == 54]
store_data.head()


# In[11]:


#Trasnactions over time for store 54
plt.figure(figsize=(11, 8))
plt.plot(store_data.index, store_data['transactions'], color='blue')
plt.xlabel('Date')
plt.ylabel('Transactions')
plt.title('Transactions Over Time')
plt.grid(True)
plt.show()


# In[12]:


#Creating training and testing data
train_size = int(len(store_data) * 0.75)
train, test = store_data['transactions'][:train_size], store_data['transactions'][train_size:]


# In[13]:


# Checking for stationarity with the ADF test
def check_stationarity(series):
    result = adfuller(series)
    return result[1] < 0.05


# In[14]:


check_stationarity(train)


# In[16]:


#ETS MODEL
ets_model = ExponentialSmoothing(train, seasonal='add', trend='add', seasonal_periods=12).fit()
ets_forecast = ets_model.forecast(len(test))


# In[17]:


#Inititalizing Parameters for Arima Model
p=1
q=1
d=1


# In[18]:


#ARIMA MODEL
arima_model = ARIMA(train, order=(p, d, q)).fit()
arima_forecast = arima_model.forecast(len(test))


# In[20]:


#Root Mean Squared errror
ets_rmse = np.sqrt(mean_squared_error(test, ets_forecast))
arima_rmse = np.sqrt(mean_squared_error(test, arima_forecast))
print(f"ETS Model RMSE: {ets_rmse}")
print(f"ARIMA Model RMSE: {arima_rmse}")


# In[21]:


#plotting actual vs predicted
plt.figure(figsize=(12, 6))
plt.plot(test.index, test, label='Actual Transactions', color='blue')
plt.plot(test.index, ets_forecast, label='ETS Forecast', linestyle='--', color='green')
plt.plot(test.index, arima_forecast, label='ARIMA Forecast', linestyle='--', color='red')
plt.xlabel('Date')
plt.ylabel('Transactions')
plt.title('Actual vs Forecasted Transactions')
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:




