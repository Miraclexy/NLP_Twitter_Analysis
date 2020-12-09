#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 13:58:23 2020

@author: yi
"""

from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
from tweepy import API 
from tweepy import RateLimitError
import tweepy
import json
import nltk
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from textblob import TextBlob
import re
import time


'''
Download tweets through Twitter API, clean tweets and get sentiments
'''
class TwitterClient(object):
    def __init__(self): 
        # consumer key, consumer secret, access token, access secret.
        ckey = ''
        csecret = ''
        atoken = ''
        asecret = ''
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
        if False:
            nltk.download('stopwords')
            nltk.download('wordnet')
            nltk.download('averaged_perceptron_tagger')
            nltk.download('punkt')
        stop_words = nltk.corpus.stopwords.words('english')
        stop_words.remove('no')
        stop_words.remove('not')
        stop_words.remove('nor')      
        for w in ['!',',','.','?','-s','-ly','</s>','s',"'s",'``',')','(',':','--',';','\n','\\',"'ll'","'d","'re","'ve",'-',"''",'*',"'","`",'&']:
            stop_words.append(w)
        
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
        
        words = re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split()
        sentence = ' '.join(word for word in [lower(w) for w in words if w not in stop_words])
        sentence = lemmatize_sentence(sentence)
        
        return sentence
    
    def get_sentiment(self, tweet, cleaned = False): 
        if not cleaned: tweet = self.clean_tweet(tweet)
        analysis = TextBlob(tweet) 
        if analysis.sentiment.polarity > 0: 
            return 'positive'
        elif analysis.sentiment.polarity == 0: 
            return 'neutral'
        else: 
            return 'negative'
        
    def download_tweets(self, query, count, until, max_id = None): 
        def helper(tweets):
            df = pd.DataFrame(data=[tweet.text for tweet in tweets], columns=['text'])
            df['id'] = np.array([tweet.id for tweet in tweets])
            df['createdate'] = np.array([tweet.created_at for tweet in tweets])
            df['source'] = np.array([tweet.source for tweet in tweets])
            df['likes'] = np.array([tweet.favorite_count for tweet in tweets])
            df['retweets'] = np.array([tweet.retweet_count for tweet in tweets])
            df['followers'] = np.array([tweet.user.followers_count for tweet in tweets])
            df['text'] = df.text.apply(self.clean_tweet)   
            df['sentiment'] = df.text.apply(self.get_sentiment,True)
            return df
        # api.user_timeline(screen_name, count, page) 
        tweets = self.api.search(q = query + '-filter:retweets', lang = 'en', count = count, until = until, max_id = max_id, result_type = "recent")
        stopped_id = tweets[-1].id
        return helper(tweets), stopped_id


def main(ticker, full):
    # can download up to 100 tweets every 15 minutes
    Tweets = pd.DataFrame()
    num = 0
    max_id = None
    while True:
        try:
            query = ticker + ' OR ' + full
            tweets, stopped_id = TC.download_tweets(query, count = 100, until = '2020-12-09', max_id = max_id) 
            Tweets = pd.concat([Tweets, tweets], ignore_index = True) 
            if len(tweets) < 10: 
                print('done')
                print('all 7-day tweets have been downloaded.')
                Tweets.to_csv('./tweets/' + ticker + '.csv')
                return 
            max_id = stopped_id
            num += len(tweets)            
            if num % 100 == 0:
                print('{} tweets have been downloaded.'.format(num))
        except RateLimitError:
            for i in range(15):
                time.sleep(60)
                print('slept for {} minutes'.format(i+1))
            print('can re-download now')
        except:
            print('all 7-day tweets have been downloaded.')
            Tweets.to_csv('./tweets/' + ticker + '.csv') 
            return


if __name__ == '__main__':
    TC = TwitterClient()
    querydict = {'$ABT':'Abbott Laboratories','$AIZ':'Assurant','$APD':'Air Products and Chemicals','$BKR':'Baker Hughes Company', \
                 '$BXP':'Boston Properties','$CSCO':'Cisco','$ETN':'Eaton','$FLT':'FLEETCOR', \
                 '$GS':'The Goldman Sachs Group','$GWW':'W.W. Grainger', '$HCA':'HCA Healthcare','$WM':'Waste Management'}
    
    for ticker, full in zip(querydict.keys(), querydict.values()):  
        main(ticker, full)
