#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 13:58:23 2020

@author: yi
"""


from tweepy import OAuthHandler, RateLimitError
import datetime
from tweepy import API 
import json
import nltk
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from textblob import TextBlob
import re
import time
import os

warnings.filterwarnings('ignore')


'''
Download tweets through Twitter API, clean tweets and get sentiments
'''
class TwitterClient(object):
    def __init__(self): 
        # consumer key, consumer secret, access token, access secret.
        ckey = '9bDddyyED8tGgcclHq5DMV54Y'
        csecret = 'frfrgSHMoPKDNKNGhjaJLEA5U8GiJwADAH5ChaQMXX2m7pWDHB'
        atoken = '1194484004617146373-kqrDClPjO8AHwCkXjn103BeCXP7lwH'
        asecret = '2f3rtkxGpMzJ22muSCIaSrwPHoag7rk97kgq6fwTwVdK6'
        try: 
            self.auth = OAuthHandler(ckey, csecret) 
            self.auth.set_access_token(atoken, asecret)
            self.api = API(self.auth)
        except: 
            print("Error: Authentication Failed") 


    def clean_tweet(self, tweet): 
        ''' 
        remove adjusted stop words, links, special characters; lemmatize tweet
        '''

        def get_wordnet_pos(tag):
            if tag.startswith('J'):
                return nltk.corpus.wordnet.ADJ
            elif tag.startswith('V'):
                return nltk.corpus.wordnet.VERB
            elif tag.startswith('N'):
                return nltk.corpus.wordnet.NOUN
            elif tag.startswith('R'):
                return nltk.corpus.wordnet.ADV
            else:
                return None   

        def lemmatize_sentence(sentence):
            res = []
            lemmatizer = nltk.stem.WordNetLemmatizer()
            for word, pos in nltk.pos_tag(nltk.word_tokenize(sentence)):
                wordnet_pos = get_wordnet_pos(pos) or nltk.corpus.wordnet.NOUN
                res.append(lemmatizer.lemmatize(word, pos=wordnet_pos))           
            return ' '.join(i for i in res)

        def lower(txt): return txt.lower()

        if False:
            nltk.download('stopwords')
            nltk.download('wordnet')
            nltk.download('averaged_perceptron_tagger')

        stop_words = nltk.corpus.stopwords.words('english')
        stop_words.remove('no')
        stop_words.remove('not')
        stop_words.remove('nor')
        for w in ['!', ',', '.', '?', '-s', '-ly', '</s>', 's', "'s", '``', ')', '(', ':', '--', ';', '\n', '\\', \
                  "'ll'", "'d", "'re", "'ve", '-', "''", '*', "'", "`", '&']:
            stop_words.append(w)

        words = re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split()
        sentence = ' '.join(word for word in [lower(w) for w in words if w not in stop_words])
        sentence = lemmatize_sentence(sentence)
        
        return sentence
    
    def get_sentiment(self, tweet, cleaned=False):
        if not cleaned:
            tweet = self.clean_tweet(tweet)
        analysis = TextBlob(tweet) 
        if analysis.sentiment.polarity > 0: 
            return 'positive'
        elif analysis.sentiment.polarity == 0: 
            return 'neutral'
        else: 
            return 'negative'
        
    def download_tweets(self, query, count, until=None, max_id=None):
        def helper(tweets):
            df = pd.DataFrame(data=[tweet.text for tweet in tweets], columns=['text'])
            df['id'] = np.array([tweet.id for tweet in tweets])
            df['createdate'] = np.array([tweet.created_at for tweet in tweets])
            df['source'] = np.array([tweet.source for tweet in tweets])
            df['likes'] = np.array([tweet.favorite_count for tweet in tweets])
            df['retweets'] = np.array([tweet.retweet_count for tweet in tweets])
            df['followers'] = np.array([tweet.user.followers_count for tweet in tweets])
            df['text'] = df.text.apply(self.clean_tweet)   
            df['sentiment'] = df.text.apply(self.get_sentiment, True)
            return df

        # api.user_timeline(screen_name, count, page) 
        tweets = self.api.search(q=query + '-filter:retweets', lang='en', count=count, max_id=max_id,
                                 until=until, result_type="recent")
        if len(tweets):
            stopped_id = tweets[-1].id
        else:
            stopped_id = None
        return helper(tweets), stopped_id


def get_oneday_tweets(ticker, full, idate:str):
    TC = TwitterClient()
    Tweets = pd.DataFrame()
    num = 0
    max_id = None
    sleep_time = 0
    until_date = (pd.to_datetime(idate) + datetime.timedelta(days=1)).date()
    actual_date = pd.to_datetime(idate).date()
    while True:
        try:
            query = ticker + ' OR ' + full
            # query = ticker
            tweets, stopped_id = TC.download_tweets(query, count=100, until=until_date.strftime('%Y-%m-%d'), max_id=max_id)
            tweets = tweets.drop_duplicates()
            Tweets = pd.concat([Tweets, tweets], ignore_index=True)

            if sleep_time != 0:
                sleep_time = 0
                print('can re-download now')

            if len(Tweets) == 1:
                print('No tweets found')
                return

            if Tweets['createdate'][len(Tweets)-1].day != actual_date.day:
                Tweets = Tweets[Tweets['createdate'].dt.day == actual_date.day]
                print('done')
                print(str(actual_date) + '\'s tweets have been downloaded.')
                Tweets.to_csv('./tweets/' + ticker + '_' + str(actual_date) + '.csv')
                return
            if len(tweets) <= 5:
                Tweets = Tweets[Tweets['createdate'].dt.day == actual_date.day]
                print('few data done')
                print(str(actual_date)+'\'s tweets have been downloaded.')
                Tweets.to_csv('./tweets/' + ticker + '_' + str(actual_date) +'.csv')
                return

            max_id = stopped_id
            num += len(tweets)
            if num % 100 == 0:
                print('{} tweets have been downloaded.'.format(num))
        except RateLimitError:
            time.sleep(60)
            sleep_time += 1
            print('slept for {} minutes'.format(sleep_time))

        except:
            Tweets = Tweets[Tweets['createdate'].dt.day == actual_date.day]
            print(str(actual_date) + '\'s tweets have been downloaded.')
            Tweets.to_csv('./tweets/' + ticker + '_' + str(actual_date) + '.csv', index=False)
            return


def get_multidate_tweets(ticker, full, start_date: str, end_date: str, delete=False):
    datestart = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    dateend = datetime.datetime.strptime(end_date, '%Y-%m-%d')

    datelist = []
    while datestart <= dateend:
        datelist.append(datestart.strftime('%Y-%m-%d'))
        datestart += datetime.timedelta(days=1)

    for idate in datelist:
        get_oneday_tweets(ticker, full, idate)
    print('all ' + str(len(datelist)) + ' tweets have been downloaded')

    Tweets = pd.DataFrame()

    for idate in datelist:
        filename = './tweets/' + ticker + '_' + idate + '.csv'
        tweets = pd.read_csv(filename)
        Tweets = pd.concat([Tweets, tweets], ignore_index=True)
        if delete:
            os.remove(filename)
    Tweets = Tweets.drop_duplicates()
    Tweets.to_csv('./tweets/'+ticker+'_all.csv')
    return


def main(ticker, full,unitl=None):
    # can download up to 100 tweets every 15 minutes
    TC = TwitterClient()
    Tweets = pd.DataFrame()
    num = 0
    max_id = None
    while True:
        try:
            query = ticker + ' OR ' + full
            tweets, stopped_id = TC.download_tweets(query, count=100, max_id=max_id)
            Tweets = pd.concat([Tweets, tweets], ignore_index=True)
            if len(tweets) == 0:
                print('done')
                print('all 7-day tweets have been downloaded.')
                Tweets.to_csv('./tweets/' + ticker + '.csv')
                return
            max_id = stopped_id
            num += len(tweets)
            if num % 100 == 0:
                print('{} tweets have been downloaded.'.format(num))
            if num > 1000:
                Tweets.to_csv('./tweets/' + ticker + '.csv')
        except RateLimitError:
            for i in range(15):
                time.sleep(60)
                print('slept for {} minutes'.format(i+1))
            print('can re-download now')
        except:
            print('all 7-day tweets have been downloaded.')
            Tweets.to_csv('./tweets/' + ticker + '.csv', index=False)
            return


if __name__ == '__main__':
    # query keyword
    querydict = {'$FLT': 'FleetCor'}
    start_date = '2020-11-29'
    end_date = '2020-12-08'

    # can download up to 100 tweets every 15 minutes
    for ticker, full in querydict.items():
        # main(ticker, full)
        get_multidate_tweets(ticker, full, start_date, end_date, True)
        # get_oneday_tweets(ticker, full, end_date)