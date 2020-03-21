# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 12:46:45 2020

@author: sidhant
"""

import nltk as nl
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from collections import Counter

class Greviance:
    
    def __init__(self,path):
        self.prbm={}
        self.path=path

        self.prbm={"Connectivity":{"connection","connectivity","network","connect"},
              "Internet":{"wifi","broadband","fibre","internet","5g","speed","4g"},
              "Account":{"bill","pay","charge","account","customer","detail","payment","subscribers","price"},
              "Products":{"data","service","mobile","number","tv","sim","retail","insurance","price","offer"},
              "Customer Support":{"singtelsupport","team","support"}}
        #num_grevience=
        

    def preprocess(self):
        data= pd.read_csv(self.path, index_col=False)
        for i in range(len(data["text"])):
            
        #    print(data["text"][2])
            temp=data["text"][i]
            temp=nl.word_tokenize(temp)
            
            wnl = nl.WordNetLemmatizer()
            tokens_lem = [ wnl.lemmatize(t,pos = 'v') for t in temp ]
            text_clean=" ".join(tokens_lem)
            data["text"][i]=text_clean
            
        #data.to_csv("out_lemma.csv",mode='a',header=True,index=False)
        
        datatest=data
        ######FOR finding classes
       
        for i in range(len(data["text"])):
            temp=datatest["text"][i]
            
            temp=nl.word_tokenize(temp)
            temp=[ t.lower() for t in temp ]
            stop = stopwords.words('english')
            tokens_nostop=[ t for t in temp if t not in stop ]
            text_clean=" ".join(tokens_nostop)
            datatest["text"][i]=text_clean
        return datatest
    
#Counter(" ".join(data["text"]).split()).most_common(250)

    def get_greviance(self):
        datatest=   self.preprocess()

        connectivity=0
        Internet=0
        Account=0
        Products=0
        CustomerSupport=0
        for i in range(len(datatest["text"])):
            if datatest["target_new"][i] == 0:
                temp_sentence=datatest["text"][i]
                class1=[t for t in temp_sentence.split() if t in self.prbm["Connectivity"]]
                if len(class1)!=0:
                    connectivity +=1
                class2=[t for t in temp_sentence.split() if t in self.prbm["Internet"]]
                if len(class2)!=0:
                    Internet +=1
                class3=[t for t in temp_sentence.split() if t in self.prbm["Account"]]
                if len(class3)!=0:
                    Account +=1
                class4=[t for t in temp_sentence.split() if t in self.prbm["Products"]]
                if len(class4)!=0:
                    Products +=1
                class5=[t for t in temp_sentence.split() if t in self.prbm["Customer Support"]]
                if len(class5)!=0:
                    CustomerSupport +=1
        greviance={"Connectivity":connectivity,"Internet":Internet,"Account":Account,"Products":Products,
                   "CustomerSupport":CustomerSupport
                
                }
        return greviance 


#####calling the class method
#grev= Greviance("C:/sidhant/NLP-Sentiment/out1.csv")
#
#print(grev.get_greviance())
        
        
        
        
        
