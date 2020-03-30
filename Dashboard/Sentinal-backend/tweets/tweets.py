import pandas as pd
import os
import datetime


def get_data_json(tweet_num, df):
    top_tweets = {}
    for _ in range(tweet_num):
        top_tweets[_] = {"tweet": df['text'][_],
                         "username": df['username'][_],
                         "time": df.index[_],
                         "sentiment": "positive" if df['label'][_] else "negative"}
    return top_tweets


class Tweets:
    def __init__(self, app):
        self.df = pd.read_csv(app.config["DATA"])
        self.df['timestamp'] = pd.to_datetime(
            self.df['timestamp'].apply(lambda x: datetime.datetime.strptime(x, '%d-%m-%Y %H.%M'))).dt.date
        self.df = self.df.set_index(['timestamp'])

        # Starhub
        self.dfs = pd.read_csv(app.config["STARHUB"])
        self.dfs['timestamp'] = pd.to_datetime(self.dfs['timestamp'].apply(lambda x: datetime.datetime.strptime(x,
                                                                                                                '%d-%m-%Y %H.%M'))).dt.date
        self.dfs = self.dfs.set_index(['timestamp'])

        # M1
        self.dfm = pd.read_csv(app.config["M1"])
        self.dfm['timestamp'] = pd.to_datetime(
            self.dfm['timestamp'].apply(lambda x: datetime.datetime.strptime(x, '%d-%m-%Y %H.%M'))).dt.date
        self.dfm = self.dfm.set_index(['timestamp'])

    def get_recent_tweets(self, telco, tweet_num=10):
        if telco == "SINGTEL":
            top_tweets = get_data_json(tweet_num, self.df.sort_index(ascending=False))
            return top_tweets
        elif telco == "STARHUB":
            top_tweets = get_data_json(tweet_num, self.dfs.sort_index(ascending=False))
            return top_tweets
        else:
            top_tweets = get_data_json(tweet_num, self.dfm.sort_index(ascending=False))
            return top_tweets

    def get_most_positive(self, telco, tweet_num=10):
        if telco == "SINGTEL":
            top_postive = self.df.sort_values('label', ascending=False)
            return get_data_json(10, top_postive)
        elif telco == "STARHUB":
            top_postive = self.dfs.sort_values('label', ascending=False)
            return get_data_json(10, top_postive)
        else:
            top_postive = self.dfm.sort_values('label', ascending=False)
            return get_data_json(10, top_postive)

    def get_most_negative(self, telco, tweet_num=10):
        if telco == "SINGTEL":
            top_negative = self.df.sort_values('label', ascending=True)
            return get_data_json(10, top_negative)
        elif telco == "STARHUB":
            top_negative = self.dfs.sort_values('label', ascending=True)
            return get_data_json(10, top_negative)
        else:
            top_negative = self.dfm.sort_values('label', ascending=True)
            return get_data_json(10, top_negative)