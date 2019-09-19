#!/usr/bin/env python
# coding: utf-8

# In[13]:


import requests,json,numpy as np,pandas as pd
#https://api.coinranking.com/v1/public/coin/:coin_id/history/:timeframe
#https://docs.coinranking.com/
from keras.models import Sequential
from keras.models import load_model
from keras.layers import LSTM, Dense, Activation
import time
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
import cufflinks as cf


# In[14]:


def hist_price_dl(coin_id=1,timeframe = "5y",currency = "USD"):#id1335
    '''It accepts coin_id, timeframe, and currency parameters to clean the historic coin data taken from COINRANKING.COM
    It returns a Pandas Series with daily mean values of the selected coin in which the date is set as the index'''
    r = requests.get("https://api.coinranking.com/v1/public/coin/"+str(coin_id)+"/history/"+timeframe+"?base="+currency)
    coin = json.loads(r.text)['data']['history'] #Reading in json and cleaning the irrelevant parts
    df = pd.DataFrame(coin)
    df['price'] = pd.to_numeric(df['price'])
    df['timestamp'] = pd.to_datetime(df['timestamp'],unit='ms').dt.date
    return df.groupby('timestamp').mean()['price']


# In[50]:


def price_matrix_creator(data, seq_len=30):
    '''
    It converts the series into a nested list where every item of the list contains historic prices of 30 days
    '''
    price_matrix = []
    if len(data) > (seq_len + 1):
        for index in range(len(data)-seq_len+1):
            price_matrix.append(data[index:index+seq_len]) 
    else:
        price_matrix.append(data[0:len(data)])
    return price_matrix

def normalize_windows(window_data):
    '''
    It normalizes each value to reflect the percentage changes from starting point
    '''
    normalised_data = []
    for window in window_data:
        normalised_window = [((float(p) / float(window[0])) - 1) for p in window]
        normalised_data.append(normalised_window)
    return normalised_data

def train_test_split_(price_matrix, train_size=0.9, shuffle=False, return_row=True):
    '''
    It makes a custom train test split where the last part is kept as the training set.
    '''
    price_matrix = np.array(price_matrix)
    #print(price_matrix.shape)
    row = int(round(train_size * len(price_matrix)))
    train = price_matrix[:row, :]
    if shuffle==True:
        np.random.shuffle(train)
    X_train, y_train = train[:row,:-1], train[:row,-1]
    X_test, y_test = price_matrix[row:,:-1], price_matrix[row:,-1]
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    if return_row:
        return row, X_train, y_train, X_test, y_test
    else:
        X_train, y_train, X_test, y_test

def deserializer(preds, data, train_size=0.9, train_phase=False):
    '''
    Arguments:
    preds : Predictions to be converted back to their original values
    data : It takes the data into account because the normalization was made based on the full historic data
    train_size : Only applicable when used in train_phase
    train_phase : When a train-test split is made, this should be set to True so that a cut point (row) 
    is calculated based on the train_size argument, otherwise cut point is set to 0
    
    Returns:
    A list of deserialized prediction values, original true values, and date values for plotting
    '''
    price_matrix = price_matrix_creator(data)
    arr_price_matrix = np.array(price_matrix)
    if train_phase:
        row = int(round(train_size * len(arr_price_matrix)))
    else:
        row=0
    date = data.index[row+29:] #+29/data
    date = np.reshape(date, (date.shape[0]))
    x_test = arr_price_matrix[row:,:-1]
    y_test = arr_price_matrix[row:,-1]
    actual = []
    preds_original = []
    preds = np.reshape(preds, (preds.shape[0]))
    for index in range(0, len(x_test)):#preds/X_test
        pred = (preds[index+row]+1)* x_test[index][0]
        preds_original.append(pred)
    #preds_original.append((preds[-1]+1) * y_test[-1])
    preds_original = np.array(preds_original)
    if train_phase:
        return [date, y_test, preds_original]
    else:
        import datetime
        return [date+datetime.timedelta(days=1),preds_original]
    


# In[16]:


# Not passing any argument since they are set by default
ser = hist_price_dl()

# Creating a matrix using the dataframe
price_matrix = price_matrix_creator(ser)

# Normalizing its values to fit to RNN
price_matrix = normalize_windows(price_matrix)

# Applying train-test splitting, also returning the splitting-point
row, X_train, y_train, X_test, y_test = train_test_split_(price_matrix)


# In[17]:


len(ser)


# In[18]:



# LSTM Model parameters, I chose
batch_size = 10            # Batch size (you may try different values)
epochs = 15               # Epoch (you may try different values)
seq_len = 30              # 30 sequence data (Representing the last 30 days)
loss='mean_squared_error' # Since the metric is MSE/RMSE
optimizer = 'nadam'     # rmsprop Recommended optimizer for RNN #Others to try - adam/nadam
activation = 'linear'     # Linear activation
input_shape=(None,1)      # Input dimension
output_dim = 30           # Output dimension


# In[19]:


model = Sequential()
model.add(LSTM(units=output_dim, return_sequences=True, input_shape=input_shape))
model.add(Dense(units=32,activation=activation))
model.add(LSTM(units=output_dim, return_sequences=False))
model.add(Dense(units=1,activation=activation))
model.compile(optimizer=optimizer,loss=loss)


# In[20]:


start_time = time.time()
model.fit(x=X_train,
          y=y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.05)
end_time = time.time()
processing_time = end_time - start_time


# In[21]:


model.save('coin_predictor.h5')


# In[129]:


arr = [1,2,3,4,5,6,7,8,9]
ser[-30:]


# In[22]:



#We need ser, preds, row
ser = hist_price_dl(timeframe='1y')#[1:31]
price_matrix = price_matrix_creator(ser)
X_test = normalize_windows(price_matrix)
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
X_test.shape


# In[67]:



model = load_model('coin_predictor.h5')
preds = model.predict(X_test, batch_size=2)


# In[390]:


final_pred = deserializer(preds, ser, train_size=0.9, train_phase=False)
final_pred[1][-1]


# Alternate load for multiple predictions

# In[23]:


model = load_model('coin_predictor.h5')
preds = model.predict(X_test, batch_size=5)


# In[53]:


plotlist = deserializer(preds, ser, train_size=0.8, train_phase=True)#Use train_size to change timeframe


# In[45]:


len(plotlist[0])
#plotlist[2]
#np.array(price_matrix)
#ser[364]
(preds[335] + 1) * ser[335]


# In[26]:


#for plotly
init_notebook_mode(connected=True)


# In[54]:


#To plot or not to plot
prices = pd.DataFrame({'Predictions':plotlist[2], 'Real Prices':plotlist[1]},index=plotlist[0])
iplot(prices.iplot(asFigure=True,
                   kind='scatter',
                   xTitle='Date',
                   yTitle='BTC Price',
                   title='BTC Price Predictions'))


# In[ ]:




