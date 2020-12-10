#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 13:58:23 2020

@author: Mengyao
"""

import pandas as pd
import warnings
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
import matplotlib.ticker
import seaborn as sns

warnings.filterwarnings('ignore')


def get_aggregate_df(market_data, ticker, zoom=4):
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


def get_x_axis(zoom):
    days = ['11-30','12-01', '12-02', '12-03', '12-04', '12-07', '12-08']
    zooms = []
    for i in range(int(24//zoom)):
        t = str(i*4)
        if int(t) < 10:
            t = '0'+t+':00'
        else:
            t = t + ':00'
        zooms.append(t)
    x = []
    for i in range(len(days)):
        for j in range(len(zooms)):
            x.append(days[i]+' '+zooms[j])
    return sorted(x)


def draw_aggregate_graph(code, ticker_df, x, zoom):
    y0 = ticker_df['excess_return']
    y1 = ticker_df['volatility']
    y2 = ticker_df['volume']
    if 'positive' in ticker_df.columns and 'negative' in ticker_df.columns:
        y3 = ticker_df['positive'] - ticker_df['negative']
    elif 'positive' in ticker_df.columns:
        y3 = ticker_df['positive']
    else:
        y3 = ticker_df['negative']

    sns.set_style("whitegrid")
    colors = sns.color_palette("RdBu_r")
    fig = plt.figure(figsize=(20, 15))
    gs = gridspec.GridSpec(4, 1)

    ax0 = plt.subplot(gs[0])
    line0, = ax0.plot(x, y0, color=colors[0])

    ax1 = plt.subplot(gs[1], sharex=ax0)
    line1 = ax1.bar(x, y1, color=colors[1])
    plt.setp(ax0.get_xticklabels(), visible=False)

    ax2 = plt.subplot(gs[2], sharex=ax0)
    line2 = ax2.bar(x, y2, color=colors[4])
    plt.setp(ax1.get_xticklabels(), visible=False)

    ax3 = plt.subplot(gs[3], sharex=ax0)
    line3 = ax3.bar(x, y3, color=colors[5])
    plt.setp(ax2.get_xticklabels(), visible=False)

    ax0.legend((line0, line1, line2, line3), ('excess return', 'volatility', 'volume', 'pos-neg'), loc='lower left')

    if zoom == 4:
        base = 3
    elif zoom == 1:
        base = 12
    ax0.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=base))

    plt.subplots_adjust(hspace=.0)
    # plt.show()
    plt.savefig('./figures/' + code + '.jpg')
    print(code + '\'s figure has been saved.')


def main():
    market_data = pd.read_csv('market_data.csv')
    codes = ['ABT', 'AIZ', 'APD', 'BKR', 'BXP', 'CSCO', 'ETN', 'FLT', 'GS', 'GWW', 'HCA', 'WM']
    for code in codes:
        num_tweets = len(pd.read_csv('./tweets/' + code + '.csv', index_col=0))
        if num_tweets >= 5000:
            zoom = 1
        else:
            zoom = 4

        ticker_df = get_aggregate_df(market_data, code, zoom)
        x = get_x_axis(zoom)
        draw_aggregate_graph(code, ticker_df, x, zoom)


if __name__ == '__name__':
    main()