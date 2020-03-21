# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 01:02:23 2020

@author: sidhant
"""

from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from datetime import datetime
import bert
from bert import run_classifier
from bert import optimization
from bert import tokenization
from tensorflow import keras
import os
import re

class BertModel :
    def __init__(self):
        self.BERT_MODEL_HUB = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
        self.BATCH_SIZE = 10
        self.LEARNING_RATE = 2e-5
        self.NUM_TRAIN_EPOCHS = 3.0
        # Warmup is a period of time where hte learning rate 
        # is small and gradually increases--usually helps training.
        self.WARMUP_PROPORTION = 0.1
        # Model configs
        self.SAVE_CHECKPOINTS_STEPS = 1000
        self.SAVE_SUMMARY_STEPS = 100
        self.DATA_COLUMN = 'text'   
        self.LABEL_COLUMN = 'target_new'
        self.OUTPUT_DIR = "C:/sidhant/NLP-Sentiment/eval"
        

    
    def create_tokenizer_from_hub_module(self):
      """Get the vocab file and casing info from the Hub module."""
      
      with tf.Graph().as_default():
        bert_module = hub.Module(self.BERT_MODEL_HUB)
        tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
        with tf.Session() as sess:
          vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
                                                tokenization_info["do_lower_case"]])
          
      return bert.tokenization.FullTokenizer(
          vocab_file=vocab_file, do_lower_case=do_lower_case)
      
      
    
    
    def create_model(self,is_predicting, input_ids, input_mask, segment_ids, labels,
                     num_labels):
      """Creates a classification model."""
    
      bert_module = hub.Module(
          self.BERT_MODEL_HUB,
          trainable=True)
      bert_inputs = dict(
          input_ids=input_ids,
          input_mask=input_mask,
          segment_ids=segment_ids)
      bert_outputs = bert_module(
          inputs=bert_inputs,
          signature="tokens",
          as_dict=True)
    
      # Use "pooled_output" for classification tasks on an entire sentence.
      # Use "sequence_outputs" for token-level output.
      output_layer = bert_outputs["pooled_output"]
    
      hidden_size = output_layer.shape[-1].value
    
      # Create our own layer to tune for politeness data.
      output_weights = tf.get_variable(
          "output_weights", [num_labels, hidden_size],
          initializer=tf.truncated_normal_initializer(stddev=0.02))
    
      output_bias = tf.get_variable(
          "output_bias", [num_labels], initializer=tf.zeros_initializer())
    
      with tf.variable_scope("loss"):
    
        # Dropout helps prevent overfitting
        output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
    
        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        log_probs = tf.nn.log_softmax(logits, axis=-1)
    
        # Convert labels into one-hot encoding
        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
    
        predicted_labels = tf.squeeze(tf.argmax(log_probs, axis=-1, output_type=tf.int32))
        # If we're predicting, we want predicted labels and the probabiltiies.
        if is_predicting:
          return (predicted_labels, log_probs)
    
        # If we're train/eval, compute loss between predicted and actual label
        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)
        return (loss, predicted_labels, log_probs)
    
    
    def model_fn_builder(self,num_labels, learning_rate, num_train_steps,
                         num_warmup_steps):
      """Returns `model_fn` closure for TPUEstimator."""
      def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""
    
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
    
        is_predicting = (mode == tf.estimator.ModeKeys.PREDICT)
        
        # TRAIN and EVAL
        if not is_predicting:
    
          (loss, predicted_labels, log_probs) = self.create_model(
            is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels)
    
          train_op = bert.optimization.create_optimizer(
              loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu=False)
    
          # Calculate evaluation metrics. 
          def metric_fn(label_ids, predicted_labels):
            accuracy = tf.metrics.accuracy(label_ids, predicted_labels)
            f1_score = tf.contrib.metrics.f1_score(
                label_ids,
                predicted_labels)
            auc = tf.metrics.auc(
                label_ids,
                predicted_labels)
            recall = tf.metrics.recall(
                label_ids,
                predicted_labels)
            precision = tf.metrics.precision(
                label_ids,
                predicted_labels) 
            true_pos = tf.metrics.true_positives(
                label_ids,
                predicted_labels)
            true_neg = tf.metrics.true_negatives(
                label_ids,
                predicted_labels)   
            false_pos = tf.metrics.false_positives(
                label_ids,
                predicted_labels)  
            false_neg = tf.metrics.false_negatives(
                label_ids,
                predicted_labels)
            return {
                "eval_accuracy": accuracy,
                "f1_score": f1_score,
                "auc": auc,
                "precision": precision,
                "recall": recall,
                "true_positives": true_pos,
                "true_negatives": true_neg,
                "false_positives": false_pos,
                "false_negatives": false_neg
            }
    
          eval_metrics = metric_fn(label_ids, predicted_labels)
    
          if mode == tf.estimator.ModeKeys.TRAIN:
            return tf.estimator.EstimatorSpec(mode=mode,
              loss=loss,
              train_op=train_op)
          else:
              return tf.estimator.EstimatorSpec(mode=mode,
                loss=loss,
                eval_metric_ops=eval_metrics)
        else:
          (predicted_labels, log_probs) = self.create_model(
            is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels)
    
          predictions = {
              'probabilities': log_probs,
              'labels': predicted_labels
          }
          return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    
      # Return the actual model function in the closure
      return model_fn
    
    
    
    
    
    
    
   
#    #@markdown Whether or not to clear/delete the directory and create a new one
#    DO_DELETE = False #@param {type:"boolean"}
#    #@markdown Set USE_BUCKET and BUCKET if you want to (optionally) store model output on GCP bucket.
#    USE_BUCKET = True #@param {type:"boolean"}
#    BUCKET = 'BUCKET_NAME' #@param {type:"string"}
#    
#    #if USE_BUCKET:
#     # OUTPUT_DIR = 'gs://{}/{}'.format(BUCKET, OUTPUT_DIR)
#      #from google.colab import auth
#      #auth.authenticate_user()
#    
#    if DO_DELETE:
#      try:
#        tf.gfile.DeleteRecursively(OUTPUT_DIR)
#      except:
#        # Doesn't matter if the directory didn't exist
#        pass
#    tf.gfile.MakeDirs(OUTPUT_DIR)
    
    
    
    
    
    
    
    
    
    # Download and process the dataset files.
    def download_and_load_datasets(force_download=False):
        data=pd.read_csv("./Data_F.csv")
        data.head()
        train_df = data[:2000] 
                                           
        test_df = data[2001:4000]
        return train_df, test_df
    
    
    
    
    
    def training(self):
        train, test = self.download_and_load_datasets()
        tokenizer = self.create_tokenizer_from_hub_module()
    # label_list is the list of labels, i.e. True, False or 0, 1 or 'dog', 'cat'
        label_list = [0, 1]
        
        
        
        
        train_InputExamples = train.apply(lambda x: bert.run_classifier.InputExample(guid=None, # Globally unique ID for bookkeeping, unused in this example
                                                                           text_a = x[self.DATA_COLUMN], 
                                                                           text_b = None, 
                                                                           label = x[self.LABEL_COLUMN]), axis = 1)
        
        test_InputExamples = test.apply(lambda x: bert.run_classifier.InputExample(guid=None, 
                                                                           text_a = x[self.DATA_COLUMN], 
                                                                           text_b = None, 
                                                                           label = x[self.LABEL_COLUMN]), axis = 1)
        MAX_SEQ_LENGTH = 130
        # Convert our train and test features to InputFeatures that BERT understands.
        train_features = bert.run_classifier.convert_examples_to_features(train_InputExamples, label_list, MAX_SEQ_LENGTH, tokenizer)
        test_features = bert.run_classifier.convert_examples_to_features(test_InputExamples, label_list, MAX_SEQ_LENGTH, tokenizer)
        return train_features,test_features,label_list
    
    
    
    def get_estmator(self):
        train_features,test_features,label_list=self.training()
        num_train_steps = int(len(train_features) / self.BATCH_SIZE * self.NUM_TRAIN_EPOCHS)
        num_warmup_steps = int(num_train_steps * self.WARMUP_PROPORTION)
        tokenizer = self.create_tokenizer_from_hub_module()
    
    
        run_config = tf.estimator.RunConfig(
        model_dir=self.OUTPUT_DIR,
        save_summary_steps=self.SAVE_SUMMARY_STEPS,
        save_checkpoints_steps=self.SAVE_CHECKPOINTS_STEPS)
        
        model_fn = self.model_fn_builder(
          num_labels=len(label_list),
          learning_rate=2e-5,
          num_train_steps=num_train_steps,
          num_warmup_steps=num_warmup_steps)
        
        estimator = tf.estimator.Estimator(
          model_fn=model_fn,
          config=run_config,
          params={"batch_size": self.BATCH_SIZE})
        
        return estimator , tokenizer
















