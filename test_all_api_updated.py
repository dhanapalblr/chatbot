

import os
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import re
import unicodedata
import nltk
from nltk.corpus import stopwords
import keras
from tqdm import tqdm
import pickle
from keras.models import Model
import keras.backend as K
from sklearn.metrics import confusion_matrix,f1_score,classification_report
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
import itertools
from keras.models import load_model
from sklearn.utils import shuffle
import torch
from transformers import DistilBertTokenizer, TFDistilBertModel, DistilBertConfig
from transformers import TFDistilBertForSequenceClassification
import torch.nn.functional as nnf
import time
import torch.nn as nn
import pandas as pd

loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5,epsilon=1e-08)
BERT_PATH="distilbert-base-uncased"

df_data=pd.read_excel('7data_train_cleaned1.xlsx')
num_classes_data=len(df_data.intent1.unique())
d_data= dict([(i,a) for i, a in zip(df_data.intent1, df_data.intent)])
tokenizer_data = DistilBertTokenizer.from_pretrained(BERT_PATH, do_lower_case=True)
MODEL_data = TFDistilBertForSequenceClassification.from_pretrained(BERT_PATH,num_labels=num_classes_data, from_pt=True)
MODEL_data.compile(loss=loss,optimizer=optimizer, metrics=[metric])
MODEL_data.load_weights('distillbert_7data_Model_N1.h5')

df_easre=pd.read_excel('7easre_train_cleaned1.xlsx')
num_classes_easre=len(df_easre.intent1.unique())
d_easre= dict([(i,a) for i, a in zip(df_easre.intent1, df_easre.intent)])
tokenizer_easre = DistilBertTokenizer.from_pretrained(BERT_PATH, do_lower_case=True)
MODEL_easre = TFDistilBertForSequenceClassification.from_pretrained(BERT_PATH,num_labels=num_classes_easre, from_pt=True)
MODEL_easre.compile(loss=loss,optimizer=optimizer, metrics=[metric])
MODEL_easre.load_weights('distillbert_7easre_Model_N1.h5')

df_ibgpdts=pd.read_excel('7ibgpdts_train_cleaned.xlsx')
num_classes_ibgpdts=len(df_ibgpdts.intent1.unique())
d_ibgpdts= dict([(i,a) for i, a in zip(df_ibgpdts.intent1, df_ibgpdts.intent)])
tokenizer_ibgpdts= DistilBertTokenizer.from_pretrained(BERT_PATH, do_lower_case=True)
MODEL_ibgpdts= TFDistilBertForSequenceClassification.from_pretrained(BERT_PATH,num_labels=num_classes_ibgpdts, from_pt=True)
MODEL_ibgpdts.compile(loss=loss,optimizer=optimizer, metrics=[metric])
MODEL_ibgpdts.load_weights('distillbert_7ibgpdts_Model_N.h5')

df_riskgov=pd.read_excel('7riskgov_train_cleaned1.xlsx')
num_classes_riskgov=len(df_riskgov.intent1.unique())
d_riskgov= dict([(i,a) for i, a in zip(df_riskgov.intent1, df_riskgov.intent)])
tokenizer_riskgov = DistilBertTokenizer.from_pretrained(BERT_PATH, do_lower_case=True)
MODEL_riskgov = TFDistilBertForSequenceClassification.from_pretrained(BERT_PATH,num_labels=num_classes_riskgov, from_pt=True)
MODEL_riskgov.compile(loss=loss,optimizer=optimizer, metrics=[metric])
MODEL_riskgov.load_weights('distillbert_7riskgov_Model_N1.h5')

df_lcs=pd.read_excel('7lcs_train1_cleaned.xlsx' )
num_classes_lcs=len(df_lcs.intent1.unique())
d_lcs= dict([(i,a) for i, a in zip(df_lcs.intent1, df_lcs.intent)])
tokenizer_lcs = DistilBertTokenizer.from_pretrained(BERT_PATH, do_lower_case=True)
MODEL_lcs = TFDistilBertForSequenceClassification.from_pretrained(BERT_PATH,num_labels=num_classes_lcs, from_pt=True)
MODEL_lcs.compile(loss=loss,optimizer=optimizer, metrics=[metric])
MODEL_lcs.load_weights('distillbert_7lcs_Model_N1.h5')

df_iris=pd.read_excel('7iris_train_cleaned.xlsx')
num_classes_iris=len(df_iris.intent1.unique())
d_iris= dict([(i,a) for i, a in zip(df_iris.intent1, df_iris.intent)])
tokenizer_iris = DistilBertTokenizer.from_pretrained(BERT_PATH, do_lower_case=True)
MODEL_iris = TFDistilBertForSequenceClassification.from_pretrained(BERT_PATH,num_labels=num_classes_iris, from_pt=True)
MODEL_iris.compile(loss=loss,optimizer=optimizer, metrics=[metric])
MODEL_iris.load_weights('distillbert_7iris_Model_N.h5')

df_hiri=pd.read_excel('7hiri_train_cleaned.xlsx')
num_classes_hiri=len(df_hiri.intent1.unique())
d_hiri= dict([(i,a) for i, a in zip(df_hiri.intent1, df_hiri.intent)])
tokenizer_hiri = DistilBertTokenizer.from_pretrained(BERT_PATH, do_lower_case=True)
MODEL_hiri = TFDistilBertForSequenceClassification.from_pretrained(BERT_PATH,num_labels=num_classes_hiri, from_pt=True)
MODEL_hiri.compile(loss=loss,optimizer=optimizer, metrics=[metric])
MODEL_hiri.load_weights('distillbert_7hiri_Model_N.h5')

def pred(text,domain):

    if domain=='data':
        tokenizer=tokenizer_data
        MODEL=MODEL_data
        d=d_data
    if domain=='easre':
        tokenizer=tokenizer_easre
        MODEL=MODEL_easre
        d=d_easre
    if domain=='ibgpdts':
        tokenizer=tokenizer_ibgpdts
        MODEL=MODEL_ibgpdts
        d=d_ibgpdts
    if domain=='riskgov':
        tokenizer=tokenizer_riskgov
        MODEL=MODEL_riskgov
        d=d_riskgov
    if domain=='hiri':
        tokenizer=tokenizer_hiri
        MODEL=MODEL_hiri
        d=d_hiri
    if domain=='iris':
        tokenizer=tokenizer_iris
        MODEL=MODEL_iris
        d=d_iris
    if domain=='lcs':
        tokenizer=tokenizer_lcs
        MODEL=MODEL_lcs
        d=d_lcs
        

    DistilBert_inp=tokenizer.encode_plus(text,add_special_tokens = True,max_length =64,pad_to_max_length = True,return_attention_mask = True)
    input_ids=DistilBert_inp['input_ids']
    attention_masks=DistilBert_inp['attention_mask']
    input_ids=np.asarray(input_ids).reshape(-1,64)
    attention_masks=np.array(attention_masks).reshape(-1,64)
    outputs = MODEL.predict([input_ids,attention_masks],batch_size=32)

    prob = nnf.softmax(torch.tensor(outputs[0]),dim=1)
    top_p, top_class = prob.topk(3, dim = 1)

    return [ {"intent":d[int(top_class[0][0])],
         "confidence":float(top_p[0][0]),
         "domain":domain},
       {"intent":d[int(top_class[0][1])],
        "confidence":float(top_p[0][1]),
        "domain":domain},
       {"intent":d[int(top_class[0][2])],
        "confidence":float(top_p[0][2]),
        "domain":domain}]

tests=['7riskgov_test_cleaned1.xlsx',
 '7data_test_cleaned1.xlsx',
 '7ibgpdts_test_cleaned.xlsx',
 '7easre_test_cleaned.xlsx',
 '7lcs_test1_cleaned.xlsx',
 '7iris_test_cleaned.xlsx',
 '7hiri_test_cleaned.xlsx']
test_data=[]
for i in tests:
  name=i.split('_')[0]
  df=pd.read_excel(i)
  df['test_data_name']=name
  test_data.append(df)

test_data=pd.concat(test_data)


from tqdm import tqdm
out=pd.DataFrame()
acc={'accuracy':[],'domain':[]}
for j in ['data','easre','ibgpdts','riskgov','iris','hiri','lcs']:
  test=test_data[test_data['test_data_name']=='7'+j]
  out=pd.DataFrame()
  print("starting prediction for {}".format(j))
  for l in tqdm(range(len(test.iloc[0:100]))):

    i=test.iloc[l]
    k=pd.DataFrame(pred(i['utterance'],j)).iloc[0:1]
    k['utterance']=i['utterance']
    k.rename(columns={'intent':'intent_predicted'},inplace=True)
    k['intent']=i['intent']
    k['compare']=k['intent']==k['intent_predicted']
    out=pd.concat([out,k])


  print("completed prediction for {}. Saving the file..".format(j))
  out.to_csv('All_models_test_output_{}.csv'.format(j),index=None)
  try:
    acc['accuracy'].append(out['compare'].value_counts()[True]/len(out)*100)
    acc['domain'].append(out['domain'].values[0])
  except:
    pass
pd.DataFrame(acc).to_csv('Accuracy.csv',index=None)
