# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 01:44:13 2020

@author: sidhant
"""
from bert import run_classifier

from bert_model import BertModel

class Prediction:
    def __init__(self):
        self.model_class=BertModel()
        self.estimator,self.tokenizer=self.model_class.get_estmator()
    
    def getPrediction(self,in_sentences):
        labels = ["Negative", "Positive"]
        label_list = [0, 1] 
        MAX_SEQ_LENGTH = 130 
        input_examples = [run_classifier.InputExample(guid="", text_a = x, text_b = None, label = 0) for x in in_sentences]
        input_features = run_classifier.convert_examples_to_features(input_examples, label_list, MAX_SEQ_LENGTH, self.tokenizer)
        predict_input_fn = run_classifier.input_fn_builder(features=input_features, seq_length=MAX_SEQ_LENGTH, is_training=False, drop_remainder=False)
        predictions = self.estimator.predict(predict_input_fn)
        return [(sentence, labels[prediction['labels']]) for sentence, prediction in zip(in_sentences, predictions)]
    
    
    def get_prediction(self,sentence_list):
#        pred_sentences = [
#          "Anyway, thanks for the kind reply. Btw, I am still a SingTel subscriber",
#          "Why is internet so thrash today @Singtel" 
#        ]        
        
        predictions = self.getPrediction(sentence_list)
        return predictions
    
    
