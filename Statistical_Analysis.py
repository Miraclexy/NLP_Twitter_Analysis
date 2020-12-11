#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 23:56:23 2020

@author: sunyingchao
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import statsmodels.api as sm

import warnings
warnings.filterwarnings('ignore')

def get_market_data(ticker):
    df = pd.read_csv('market_data.csv')
    spy = pd.read_csv('SPY.csv')
    spy = spy.rename(columns={'PRICE':'SPYPRICE'})
    spy = spy.drop(columns = ['SYM_ROOT'])
    def cut(s):
        return s.split(':')[0]
    df['TIME_M'] = df['TIME_M'].apply(lambda x:cut(x))
    spy['TIME_M'] = spy['TIME_M'].apply(lambda x:cut(x))
    
    def get_return(x):
        x['RET'] = np.log(x.iloc[-1,-1])/np.log(x.iloc[0,-1])
        return x
    
    ret1 = df.groupby(['DATE','SYM_ROOT']).apply(get_return)
    ret2 = spy.groupby(['DATE']).apply(get_return)
    ret2 = ret2.groupby(['DATE']).agg({'RET':'mean'})
    ret2 = ret2.reset_index()
    
    data = ret1[ret1['SYM_ROOT']==ticker]
    data = data.groupby(['DATE']).agg(RET1 = ('RET', 'mean'), SIZE = ('SIZE', 'sum'), MAXPRICE = ('PRICE', 'max'), MINPRICE = ('PRICE', 'min'))
    data = data.reset_index()
    data['VOL'] = 2*(data['MAXPRICE']-data['MINPRICE'])/(data['MAXPRICE']+data['MINPRICE'])
    data = data.merge(right=ret2, on=['DATE'],how='inner')
    data['ExRET'] = data['RET1']- data['RET']
    data['DATE'] = data['DATE'].apply(lambda x: str(x))
    #get sentiment_data
    sdata = pd.read_csv('./tweets/' + ticker + '.csv', index_col=0)
    sdata['DATE'] = sdata['createdate'].apply(lambda x: x[:4] + x[5:7] + x[8:10])
    sdata = sdata.groupby(['DATE', 'sentiment']).size().unstack()
    data = data.merge(sdata, on='DATE', how='inner')
    for x in ['positive', 'negative', 'neutral']:
        if x not in data.columns.values:
            data[x] = np.nan
    data = data.fillna({'negative':0.0, 'neutral':0.0, 'positive':0.0})
    data = data.dropna()
    return data




if __name__ == '__main__':
    codes = ['ABT', 'AIZ', 'APD', 'BKR', 'BXP', 'CSCO', 'ETN', 'FLT', 'GS', 'GWW', 'HCA', 'WM']
    for code in codes:
        try:
            ticker_df = get_market_data(code)
            ticker_df['G'] = ticker_df['positive'].shift(1)
            ticker_df['B'] = ticker_df['negative'].shift(1)
            ticker_df['ER1'] = ticker_df['ExRET'].shift(1)
            ticker_df['ER2'] = ticker_df['ExRET'].shift(2)
            ticker_df['VOL1'] = ticker_df['VOL'].shift(1)
            ticker_df['VOL2'] = ticker_df['VOL'].shift(2)
            ticker_df['volume1'] = ticker_df['SIZE'].shift(1)
            ticker_df['volume2'] = ticker_df['SIZE'].shift(2)
            ticker_df = ticker_df.dropna()
            ticker_df['SR'] = (ticker_df['G']-ticker_df['B'])/(ticker_df['G']+ticker_df['B'])
            ticker_df['const'] = 1
            
            reg1 = sm.OLS(endog=ticker_df['ExRET'], exog=ticker_df[['ER1', 'ER2', 'G', 'B', 'SR', 'const']], missing='drop')
            results = reg1.fit()
            with open('./regression results/'+ code +' excess return.txt', 'w') as fh1:
                fh1.write(results.summary().as_text())
            reg2 = sm.OLS(endog=ticker_df['SIZE'], exog=ticker_df[['volume1', 'volume2', 'G', 'B', 'const']], missing='drop')
            results2 = reg2.fit()
            with open('./regression results/'+ code +' volume.txt', 'w') as fh2:
                fh2.write(results2.summary().as_text())
            reg3 = sm.OLS(endog=ticker_df['VOL'], exog=ticker_df[['VOL1', 'VOL2', 'G', 'B', 'const']], missing='drop')
            results3 = reg3.fit()
            with open('./regression results/'+ code +' volatility.txt', 'w') as fh3:
                fh3.write(results3.summary().as_text())
        except:
            print(code)
        
        
    