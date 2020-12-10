#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 17:00:36 2020

@author: yi
"""

'''
Depicts the distribution of values of the relative sentiment obtained from Twitter
Complete statistical analysis of the relationship between excess returns, 
volume and volatility with Twitter Sentiment across the sample of 12 individual securities 
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import glob
import seaborn as sns


def get_file(path = './tweets/*'):
    filelist = glob.glob('./tweets/*')
    Tweets = dict()
    for f in filelist:
        name = re.search('\/\$\w*', f).group()[2:]
        tweets = pd.read_csv(f, index_col=0)
        tweets['createdate'] = tweets['createdate'].apply(lambda x: x[:10])
        Tweets[name] = tweets.drop_duplicates()
    return Tweets

def cal_sentiment(Tweets):
    '''
    calculate relative sentiment
    '''
    sentiments = dict()        
    for comp, tweet in zip(Tweets.keys(), Tweets.values()):
        sentiments[comp] = pd.DataFrame(index = tweet[['createdate', 'sentiment']].groupby('createdate').agg({'sentiment': 'count'}).index)
        sentiments[comp]['#tweets'] =  tweet[['createdate', 'sentiment']].groupby('createdate').agg({'sentiment': 'count'})
        try:
            pos = tweet[['createdate', 'sentiment']].groupby(['createdate','sentiment']).agg({'sentiment': 'count'}).loc[(slice(None),'positive'),:].unstack()
            sentiments[comp]['#pos'] = pos
        except: sentiments[comp]['#pos'] = 0
        try:
            neg = tweet[['createdate', 'sentiment']].groupby(['createdate','sentiment']).agg({'sentiment': 'count'}).loc[(slice(None),'negative'),:].unstack()
            sentiments[comp]['#neg'] = neg
        except: sentiments[comp]['#neg'] = 0  
        try:
            neu = tweet[['createdate', 'sentiment']].groupby(['createdate','sentiment']).agg({'sentiment': 'count'}).loc[(slice(None),'neutral'),:].unstack()
            sentiments[comp]['#neu'] = neu
        except: sentiments[comp]['#neu'] = 0
        sentiments[comp] = sentiments[comp].fillna(0)                 
        sentiments[comp]['relative_sentiment'] = (sentiments[comp]['#pos'] - sentiments[comp]['#neg']) / (sentiments[comp]['#pos'] + sentiments[comp]['#neg'])
        # if relative sentiment = nan, no positive or negative only neutral
        sentiments[comp] = sentiments[comp].fillna(0)
    return sentiments
    
def plot_dailysenti(sentiments):
    '''plot everyday's relative sentiment
    '''
    plt.figure(figsize = (20, 15))
    k = 1
    for i in sentiments.keys():
        plt.subplot(4,3,k)
        plt.plot(sentiments[i][['relative_sentiment']], label = i)
        plt.tick_params(axis='x', labelsize=8)
        plt.xticks(rotation=-25) 
        plt.legend(loc = 'best')
        k += 1
    plt.show()

def plot_sentidist(sentiments):
    '''
    plot Kernal density of relative sentiment for each company
    '''
    plt.figure(figsize = (20, 15))
    k = 1
    for i in sentiments.keys():
        plt.subplot(4,3,k)
        sns.kdeplot(sentiments[i]['relative_sentiment'], shade = True, label = i)
        plt.legend(loc = 'best')
        k += 1
    plt.savefig('kernal density.png')



Tweets = get_file()        
sentiments = cal_sentiment(Tweets)
plot_dailysenti(sentiments)
# replicate Figure 2
plot_sentidist(sentiments)













''' 
def get_stockdata(Tweets):
    stockdata = dict()
    for comp, tweet in zip(Tweets.keys(), Tweets.values()):
        startdate = sorted(tweet['createdate'])[0]
        enddate = sorted(tweet['createdate'])[-1]
        data = web.DataReader(name=comp, data_source='yahoo', start=startdate+timedelta(-1), end=enddate)
        benchmark = web.DataReader(name='^GSPC', data_source='yahoo', start=startdate+timedelta(-1), end=enddate)
        stockdata[comp] = pd.DaraFrame(index = data.index)
        stockdata[comp]['excess_ret'] = data['']
        
'''




        



