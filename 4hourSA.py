#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 13:44:45 2020

@author: sunyingchao
"""
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import statsmodels.api as sm
from Generate_figures import get_aggregate_df


if __name__ == '__main__':
    market_data = pd.read_csv('market_data.csv')
    codes = ['ABT', 'AIZ', 'APD', 'BKR', 'BXP', 'CSCO', 'ETN', 'FLT', 'GS', 'GWW', 'HCA', 'WM']
    for code in codes:
        try:
            ticker_df = get_aggregate_df(market_data, code, zoom=4)
            for x in ['positive', 'negative', 'neutral']:
                if x not in ticker_df.columns.values:
                    ticker_df[x] = np.nan
            ticker_df = ticker_df.fillna({'negative':0.0, 'neutral':0.0, 'positive':0.0})

            ticker_df = ticker_df.dropna()
            #normalization
            ticker_df['excess_return'] = (ticker_df['excess_return']-np.mean(ticker_df['excess_return']))/np.std(ticker_df['excess_return'])
            ticker_df['excess_return'] = (ticker_df['volatility']-np.mean(ticker_df['volatility']))/np.std(ticker_df['volatility'])
            ticker_df['volume'] = (ticker_df['volume']-np.mean(ticker_df['volume']))/np.std(ticker_df['volume'])


            ticker_df['G'] = ticker_df['positive'].shift(1)
            ticker_df['B'] = ticker_df['negative'].shift(1)
            ticker_df['ER1'] = ticker_df['excess_return'].shift(1)
            ticker_df['ER2'] = ticker_df['excess_return'].shift(2)
            ticker_df['VOL1'] = ticker_df['volatility'].shift(1)
            ticker_df['VOL2'] = ticker_df['volatility'].shift(2)
            ticker_df['volume1'] = ticker_df['volume'].shift(1)
            ticker_df['volume2'] = ticker_df['volume'].shift(2)
            ticker_df = ticker_df.dropna()
            ticker_df['SR'] = (ticker_df['G']-ticker_df['B'])/(ticker_df['G']+ticker_df['B'])
            ticker_df['const'] = 1
            
            reg1 = sm.OLS(endog=ticker_df['excess_return'], exog=ticker_df[['ER1', 'ER2', 'G', 'B', 'SR', 'const']], missing='drop')
            results = reg1.fit()
            with open('./regression results/'+ code +' excess return.txt', 'w') as fh1:
                fh1.write(results.summary().as_text())
            reg2 = sm.OLS(endog=ticker_df['volume'], exog=ticker_df[['volume1', 'volume2', 'G', 'B', 'const']], missing='drop')
            results2 = reg2.fit()
            with open('./regression results/'+ code +' volume.txt', 'w') as fh2:
                fh2.write(results2.summary().as_text())
            reg3 = sm.OLS(endog=ticker_df['volatility'], exog=ticker_df[['VOL1', 'VOL2', 'G', 'B', 'const']], missing='drop')
            results3 = reg3.fit()
            with open('./regression results/'+ code +' volatility.txt', 'w') as fh3:
                fh3.write(results3.summary().as_text())
        except:
            print(code)
