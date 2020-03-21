# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 17:26:56 2020

@author: sidhant
"""

import requests
import pandas as pd
import numpy as np
import json
import re
key="AIzaSyBWRLDVYtpArqJouuXjZlJxk-CTWau1xlE"
address="https://language.googleapis.com/v1beta2/documents:analyzeSentiment?key="
url= address + key

temp=[]
#senti_score=pd.DataFrame(columns=['score'])
def abc(x):
    if x==0 :
        return False
    elif x in range(8000,12000):
        
        return False
    else:
        
        return True



datas=pd.read_csv("C:/sidhant/NLP-Sentiment/DECCC.csv", error_bad_lines=False,skiprows=lambda x:abc(x))
#print(datas['text'].isna().value_counts())
#print(len(datas))
#print(data_new)


data_new=pd.DataFrame(datas['text'],columns=['text'])
#data_new['text']=datas['text']



processed_data=data_new.dropna(axis=0)
processed_data= processed_data.reset_index(drop=True)
#print(processed_data.text[134])
print(processed_data)
#print(len(processed_data))
##############remove special character and url
for indx in range(len(processed_data.text)):
#    print(processed_data.text[indx])
    processed_data.text[indx]=re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', processed_data.text[indx])
    for k in processed_data.text[indx]:
        if not re.match(r"[a-zA-Z0-9@]",k):
            if k!=" ":
                processed_data.text[indx]=processed_data.text[indx].replace(k,"")
    
                

for item in processed_data.text:
    if item != "":

        document={
                         'language': 'en-us',
                         'type': 'PLAIN_TEXT',
                         'content':item 
                    }
        
        inputs={
         'document':document ,
         'encodingType': 'UTF8'
                }
        
        #payload=json.dumps(inputs)
        
        json_data=requests.post(url, headers={"Content-Type":"application/json"},
                          data=json.dumps(inputs))
        
    #    print(json_data.json())
        
        temp.append(json_data.json()['documentSentiment']['score'])


print(temp)
np.savetxt("foo.csv", temp, delimiter=",")
processed_data['target']=temp

processed_data.to_csv("out_newlabels.csv",mode='a',header=False)



