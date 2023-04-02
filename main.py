#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 11:16:03 2023
@author: hanzaz ali
"""

#import relevant librariesc

import streamlit as st
import pickle
import numpy as np
import pandas as pd

import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, TimeDistributed
import matplotlib.pyplot as plt
import plotly_express as px
import yfinance as yf


#import dataset 
df = yf.download('EURUSD=X', period="2y", interval = "1h")


df.drop(['Adj Close', 'Volume'], axis = 1, inplace = True)

# features engineering

# TYPICAL PRICE
df['TP'] = (df['Open'] + df['High'] + df['Low'])/3

# standard deviation of the moving average of 20 periods
df['Std'] = df['TP'].rolling(10).std(ddof=0)

#moving average of the typical price
df['MA-TP'] = df['TP'].rolling(10).mean()

# upper and lower bolinger bands
df['BOLU'] = df['MA-TP'] + 2 * df['Std']
df['BOLD'] = df['MA-TP'] - 2 * df['Std']

df.dropna(axis = 0, inplace = True)

df = df[['MA-TP', 'BOLU', 'BOLD', 'Close']]



#create the starting date for predictions

# create a starting date
# create a starting date
from datetime import datetime, timedelta

today_start = datetime.today().today() - timedelta(hours = datetime.today().now().hour +  datetime.today().now().minute/60 +  datetime.today().now().second/3600)

if datetime.today().date().weekday() == 5:
    start = today_start + timedelta(days = 2)

elif datetime.today().date().weekday() == 6:
    start = today_start + timedelta(days = 1)
    
    
else:
    #start = today_start - timedelta(days = datetime.today().date().weekday())
    start = today_start


#features extraction function
def preprocess_multistep_lstm(sequence, n_steps_in, n_steps_out, n_features):
    
    X, y = list(), list()
    
    for i in range(len(sequence)):
        
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        
        # check if we are beyond the sequence
        if out_end_ix > len(sequence):
            break
            
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix, -1:]
        X.append(seq_x)
        y.append(seq_y)

    X = np.array(X)
    y = np.array(y)

    X = X.reshape((X.shape[0], X.shape[1], n_features))
    
    return X, y




########## hour based model #################

#create prediction features
# Number of days into the future we want to predict
n_steps_out = 2*24

# choose the number of days on which to base our predictions 
nb_days = 10*24

n_features = df.shape[1]

inputs, target = preprocess_multistep_lstm(df.tail(12*24).to_numpy(), nb_days, n_steps_out, n_features)


#load the model
def load_model():
    with open('Model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()


#make predictions
predictions = pd.DataFrame(model.predict(inputs).T, 
                          index = [start + timedelta(hours = i) for i in range(n_steps_out)],
                          columns = ['pred Close'])


########## daily based average calculation #################
# daily base chart
daily_avg = predictions.groupby(predictions.index.date).mean()

#fig_d, ax = plt.subplots()
#ax.plot(daily_avg.index, daily_avg['pred Close'])
#ax.set_xlabel('date')
#ax.set_ylabel('price')

#plt.xticks(rotation = 90)
#plt.show()




# hour base chart
fig_h, ax = plt.subplots()
ax.plot(predictions.index, predictions['pred Close'])
ax.set_xlabel('date')
ax.set_ylabel('price')

plt.xticks(rotation = 90)
plt.show()


# hour base chart combined with the last 15 days data
fig2, ax = plt.subplots(figsize=(10, 5))
ax.plot(df[df.index > datetime.today() - timedelta(days = 15)].index, 
        df[df.index > datetime.today() - timedelta(days = 15)]['Close'])
ax.plot(predictions.index, predictions['pred Close'])
ax.set_xlabel('date')
ax.set_ylabel('price')

plt.xticks(rotation = 90)
plt.show()


#historical price

fig3 = px.line(df, x=df.index, y="Close")
#fig3.show()


def main():
    
    st.title('EUR/USD weekly predictions')
    
    st.header('last 15 days and next five days predictions')
    st.plotly_chart(fig2)
    
    
    st.header('weekly predictions display hour base')
    st.plotly_chart(fig_h)
    
    st.header('weekly predictions table daily average')
    st.write(round(daily_avg, 6))
    
    
    st.header('historical price')
    st.plotly_chart(fig3)
    
     
    
if __name__ == '__main__':
    main()
    
    
    