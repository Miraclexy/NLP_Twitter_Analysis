#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 23:56:23 2020

@author: sunyingchao
"""

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time



def get_market_data():
    df = pd.read_csv('marketdata.csv')
    spy = pd.read_csv('SPY.csv')
    spy = spy.rename(columns={'PRICE':'SPYPRICE'})
    spy = spy.drop(columns = ['SYM_ROOT'])
    def cut(s):
        return s.split(':')[0]
    df['TIME_M'] = df['TIME_M'].apply(lambda x:cut(x))
    spy['TIME_M'] = spy['TIME_M'].apply(lambda x:cut(x))
    
    def get_return(x):
        x['RET'] = x.iloc[-1,-1]/x.iloc[0,-1]
        return x
    
    ret1 = df.groupby(['DATE','TIME_M','SYM_ROOT']).apply(get_return)
    ret2 = spy.groupby(['DATE','TIME_M']).apply(get_return)
    ret2 = ret2.groupby(['DATE','TIME_M']).agg({'RET':'mean'})
    ret2 = ret2.reset_index()
    for i in set(df['SYM_ROOT'].values):
        data = ret1[ret1['SYM_ROOT']==i]
        data = data.groupby(['DATE','TIME_M']).agg(RET1 = ('RET', 'mean'), SIZE = ('SIZE', 'sum'), MAXPRICE = ('PRICE', 'max'), MINPRICE = ('PRICE', 'min'))
        data = data.reset_index()
        data['VOL'] = 2*(data['MAXPRICE']-data['MINPRICE'])/(data['MAXPRICE']+data['MINPRICE'])
        data = data.merge(right=ret2, on=['DATE','TIME_M'],how='inner')
        data['ExRET'] = data['RET1']- data['RET']
        data.to_csv('./marketdata/' + i + '.csv')

if __name__ == '__main__':
    get_market_data()
    