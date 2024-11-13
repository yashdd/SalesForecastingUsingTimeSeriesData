#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.exponential_smoothing.ets import ETSModel as ets
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA


# In[2]:


train_data = pd.read_csv("train.csv")
holiday_data = pd.read_csv("holidays_events.csv")
oil_data = pd.read_csv("oil.csv")
transactions_data = pd.read_csv("transactions.csv")
stores_data = pd.read_csv("stores.csv")


# In[3]:


#Merging the data
train_data = train_data.merge(transactions_data, on=['date','store_nbr'],how='left')


# In[4]:


train_data = train_data.merge(oil_data, on='date',how='left')


# In[5]:


train_data = train_data.merge(holiday_data, on='date',how='left')


# In[6]:


train_data = train_data.merge(stores_data, on='store_nbr',how='left')


# In[7]:


train_data


# In[8]:


train_data['date'] = pd.to_datetime(train_data['date'])
train_data['year'] = train_data['date'].dt.year
train_data['month'] = train_data['date'].dt.month
train_data['day'] = train_data['date'].dt.day


# In[9]:


train_data


# In[10]:


from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
train_data['family'] = label_encoder.fit_transform(train_data['family'])


# In[11]:


features_imp = ['store_nbr','family','onpromotion','year','month','day']


# In[12]:


X = train_data[features_imp]
y = train_data['sales']


# In[13]:


get_ipython().system('pip install xgboost')


# In[14]:


import xgboost as xgb
from sklearn.model_selection import train_test_split

X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.25, random_state=42)


# In[15]:


model = xgb.XGBRegressor(objective='reg:squarederror', max_depth=5,learning_rate = 0.1,n_estimaators=100)


# In[16]:


model.fit(X_train,y_train)


# In[17]:


y_pred = model.predict(X_val)


# In[18]:


from sklearn.metrics import mean_squared_error

print('RMSE:', np.sqrt(mean_squared_error(y_val, y_pred)))


# In[19]:


test_data = pd.read_csv("test.csv")


# In[20]:


test_data['date'] = pd.to_datetime(test_data['date'])
test_data['year'] = test_data['date'].dt.year
test_data['month'] = test_data['date'].dt.month
test_data['day'] = test_data['date'].dt.day


# In[21]:


test_data['family'] = label_encoder.transform(test_data['family'])


# In[22]:


test_data.fillna(0,inplace=True)


# In[23]:


X_test = test_data[features_imp]
test_data['sales'] = model.predict(X_test)


# In[24]:


test_data


# In[ ]:




