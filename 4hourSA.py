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
import matplotlib.pyplot as plt
import time
import statsmodels.api as sm

def get_aggregate_df(market_data, ticker, zoom=1):
    def get_timezoom(x, zoom):
        if x[1] != ':':
            t = int(x[:2])
        else:
            t = int(x[0])
        return str(t // zoom)

    ticker_market = market_data[market_data['SYM_ROOT'] == ticker]
    ticker_market['timezoom'] = ticker_market['TIME_M'].apply(lambda x: get_timezoom(x, zoom))
    ticker_market['createdate'] = ticker_market['DATE'].apply(lambda x: str(x)) + ' ' + ticker_market['timezoom']
    ticker_market = ticker_market[['SIZE', 'PRICE', 'createdate']]

    ticker_size = ticker_market.groupby(['createdate']).sum()['SIZE']
    ticker_high = ticker_market.groupby(['createdate']).max()['PRICE']
    ticker_low = ticker_market.groupby(['createdate']).min()['PRICE']
    ticker_open = ticker_market.groupby(['createdate'])['PRICE'].apply(lambda x: x.iloc[0])
    ticker_close = ticker_market.groupby(['createdate'])['PRICE'].apply(lambda x: x.iloc[-1])
    ticker_market_zoom = pd.DataFrame([ticker_size, ticker_high, ticker_low, ticker_open, ticker_close]).T
    ticker_market_zoom.columns = ['volume', 'high', 'low', 'open', 'close']

    ticker_market_zoom['volatility'] = 2 * (ticker_market_zoom['high'] - ticker_market_zoom['low']) / (
                ticker_market_zoom['low'] + ticker_market_zoom['high'])
    ticker_market_zoom['return'] = np.log(ticker_market_zoom['close']) / np.log(ticker_market_zoom['open'])

    spy = pd.read_csv('SPY.csv')
    spy['timezoom'] = spy['TIME_M'].apply(lambda x: get_timezoom(x, zoom))
    spy['createdate'] = spy['DATE'].apply(lambda x: str(x)) + ' ' + spy['timezoom']
    spy_zoom = pd.DataFrame(columns=['close', 'open'])
    spy_zoom['open'] = spy.groupby(['createdate'])['PRICE'].apply(lambda x: x.iloc[0])
    spy_zoom['close'] = spy.groupby(['createdate'])['PRICE'].apply(lambda x: x.iloc[-1])
    spy_zoom['return'] = np.log(spy_zoom['close']) / np.log(spy_zoom['open'])

    ticker_market_zoom['excess_return'] = ticker_market_zoom['return'] - spy_zoom['return']
    # ticker_market_zoom = ticker_market_zoom[['volume', 'volatility', 'excess_return']]

    ticker_tweet = pd.read_csv('./tweets/' + ticker + '.csv', index_col=0)
    ticker_tweet['createdate'] = ticker_tweet['createdate'].apply(
        lambda x: x[:4] + x[5:7] + x[8:10] + ' ' + str(int(x[12]) // zoom))
    ticker_tweet_groupby = ticker_tweet.groupby(['createdate', 'sentiment']).size().unstack()

    ticker = ticker_market_zoom.merge(ticker_tweet_groupby, on='createdate', how='outer').iloc[3:, :]

    days = ['20201130', '20201201', '20201202', '20201203', '20201204', '20201207', '20201208']
    zooms = [i for i in range(int(24 / zoom))]
    x = []
    for day in days:
        for t in zooms:
            x.append(day + ' ' + str(t))
    empty_df = pd.DataFrame(index=x)
    empty_df.index.name = 'createdate'
    ticker = empty_df.merge(ticker, on='createdate', how='left').sort_index()
    return ticker

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