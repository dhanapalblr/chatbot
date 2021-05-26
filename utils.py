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
BERT_PATH='distilbert-base-uncased'


df_iris=pd.read_excel('train_iris_fullcleaned.xlsx')
num_classes_iris=len(df_iris.intent1.unique())
d_iris= dict([(i,a) for i, a in zip(df_iris.intent1, df_iris.intent)])
tokenizer_iris = DistilBertTokenizer.from_pretrained(BERT_PATH, do_lower_case=True)
MODEL_iris = TFDistilBertForSequenceClassification.from_pretrained(BERT_PATH,num_labels=num_classes_iris)
MODEL_iris.compile(loss=loss,optimizer=optimizer, metrics=[metric])
MODEL_iris.load_weights('distillbert_model_iris.h5')

df_hiri=pd.read_excel('train_hiri_fullcleaned.xlsx')
num_classes_hiri=len(df_hiri.intent1.unique())
d_hiri= dict([(i,a) for i, a in zip(df_hiri.intent1, df_hiri.intent)])
tokenizer_hiri = DistilBertTokenizer.from_pretrained(BERT_PATH, do_lower_case=True)
MODEL_hiri = TFDistilBertForSequenceClassification.from_pretrained(BERT_PATH,num_labels=num_classes_hiri)
MODEL_hiri.compile(loss=loss,optimizer=optimizer, metrics=[metric])
MODEL_hiri.load_weights('distillbert_model_hiri.h5')

df_lcs=pd.read_excel('train_lcs_fullcleaned.xlsx')
num_classes_lcs=len(df_lcs.intent1.unique())
d_lcs= dict([(i,a) for i, a in zip(df_lcs.intent1, df_lcs.intent)])
tokenizer_lcs = DistilBertTokenizer.from_pretrained(BERT_PATH, do_lower_case=True)
MODEL_lcs = TFDistilBertForSequenceClassification.from_pretrained(BERT_PATH,num_labels=num_classes_lcs)
MODEL_lcs.compile(loss=loss,optimizer=optimizer, metrics=[metric])
MODEL_lcs.load_weights('distillbert_model_lcs.h5')

def pred(text,domain):

#     maxindx,predictions,maxindx1,maxindx2,maxindx3,maxindx4,predictions1,predictions2,predictions3,predictions4=[],[],[],[],[],[],[],[],[],[]
#     top_p1,top_p2,top_p3,top_class1,top_class2,top_class3=[],[],[],[],[],[]

    if domain=='iris':
        tokenizer=tokenizer_iris
        MODEL=MODEL_iris
        d=d_iris
    if domain=='hiri':
        tokenizer=tokenizer_hiri
        MODEL=MODEL_hiri
        d=d_hiri
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
#     maxidx = outputs[0].argmax(axis=1)
#     maxindx.append(int(maxidx))

#     predictions.append(d[int(maxidx)])
    prob = nnf.softmax(torch.tensor(outputs[0]),dim=1)
    top_p, top_class = prob.topk(3, dim = 1)

#     top_p1.append(float(top_p[0][0]))
#     top_p2.append(float(top_p[0][1]))
#     top_p3.append(float(top_p[0][2]))
#     top_class1.append(int(top_class[0][0]))
#     top_class2.append(int(top_class[0][1]))
#     top_class3.append(int(top_class[0][2]))

#     predictions1.append(d[int(top_class[0][0])])
#     predictions2.append(d[int(top_class[0][1])])
#     predictions3.append(d[int(top_class[0][2])])

    return [ {"intent":d[int(top_class[0][0])],
         "confidence":float(top_p[0][0]),
         "domain":domain},
       {"intent":d[int(top_class[0][1])],
        "confidence":float(top_p[0][1]),
        "domain":domain},
       {"intent":d[int(top_class[0][2])],
        "confidence":float(top_p[0][2]),
        "domain":domain}]