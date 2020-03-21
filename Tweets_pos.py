# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 16:23:07 2020

@author: Akshay Sachdeva
"""

import pandas as pd

import datetime as dt

from twitterscraper import query_tweets
from test_prediction import Prediction
import re



class tweet_pos:
    def tweets_sent(date1, date2):
        
        
        lang= 'english'
        begin_date = date1
        end_date = date2
        print(begin_date)
        print(end_date)
        tweets = query_tweets("Singtel", begindate = begin_date, enddate=end_date, lang = lang  )
        df = pd.DataFrame(t.__dict__ for t in tweets)
        df.to_csv ('dataaaa.csv', index = None, header=True) 
        #Don't forget to add '.csv' at the end of the path
        data = pd.read_csv('dataaaa.csv')

        data_new=data.dropna(subset=['text'])
        data_new= data_new.reset_index(drop=True)
        for indx in range(len(data_new['text'])):
            #print(processed_data.text[indx])
            #print(indx)
            data_new.text[indx]=re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', data_new.text[indx])
            for k in data_new.text[indx]:
                if not re.match(r"[a-zA-Z0-9@]",k):
                    if k!=" ":
                        data_new.text[indx]=data_new.text[indx].replace(k,"")

        listing  = data_new['text'].tolist()
        abc = Prediction()
        ab = abc.get_prediction(listing)
        #print(ab)
        d = pd.DataFrame(ab, columns= ['text','label'])
        d.loc[d['label'] == 'Positive', 'label'] = 1 
        d.loc[d['label'] == 'Negative', 'label'] = 0
        negative=d[d['label']==0]['label'].count()
        positive=d[d['label']==1]['label'].count()
        print(positive, negative)

 
date1 = dt.date(2020,1,14)
date2 = dt.date(2020,1,15)

tweet_pos.tweets_sent(date1, date2)


      
#return negative, positive
 

#date1 = dt.date(2020,1,2)
#date2 = dt.date(2020,1,3)

#neg, pos = import_data(date1,date2)

 



