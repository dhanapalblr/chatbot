import pandas as pd
import tez
import torch 
import torch.nn as nn
import transformers
from sklearn import metrics, model_selection, preprocessing
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.utils import shuffle
import torch.nn.functional as nnf

class BERTBaseUncased(tez.Model):
    def __init__(self, num_train_steps, num_classes):
        super().__init__()
        self.tokenizer = transformers.BertTokenizer.from_pretrained(
            "/nlv/ForThiya/intentprocess/bert-large-uncased", do_lower_case=True
        )
        self.bert = transformers.BertModel.from_pretrained("/nlv/ForThiya/intentprocess/bert-large-uncased")
        self.bert_drop = nn.Dropout(0.3)
        self.out = nn.Linear(1024, num_classes)
        self.num_train_steps = num_train_steps
        self.step_scheduler_after = "batch"
    def fetch_optimizer(self):
        param_optimizer = list(self.named_parameters())
        no_decay = ["bias", "LayerNorm.bias"]
        optimizer_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.001,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        opt = AdamW(optimizer_parameters, lr=3e-5)
        return opt
    def fetch_scheduler(self):
        sch = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=0, num_training_steps=self.num_train_steps
        )
        return sch
    def loss(self, outputs, targets):
        if targets is None:
            return None
        return nn.CrossEntropyLoss()(outputs, targets)
    def monitor_metrics(self, outputs, targets):
        if targets is None:
            return {}
        outputs = torch.argmax(outputs, dim=1).cpu().detach().numpy()
        targets = targets.cpu().detach().numpy()
        accuracy = metrics.accuracy_score(targets, outputs)
        return {"accuracy": accuracy}
    def forward(self, ids, mask, token_type_ids, targets=None):
        _, o_2 = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        b_o = self.bert_drop(o_2)
        output = self.out(b_o)
        loss = self.loss(output, targets)
        acc = self.monitor_metrics(output, targets)
        return output, loss, acc


import transformers
DEVICE = "cuda"
MAX_LEN = 64
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 10
BERT_PATH = "/nlv/ForThiya/intentprocess/bert-large-uncased"
MODEL_PATH = "model_fold01.bin"
TRAINING_FILE = "/root/docker_data/train.csv"
TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_PATH, do_lower_case=True)

# In[23]:

import time
import torch.nn as nn
import pandas as pd
MODEL = BERTBaseUncased(4095,500)
for i in range(0,1):
    MODEL.load_state_dict(torch.load("model_fold0.bin"))
    MODEL.to(DEVICE)
    MODEL.eval()
    # DEVICE = config.DEVICE
    # PREDICTION_DICT = dict()
    tokenizer = TOKENIZER
    max_len = MAX_LEN
    # sent_csv=pd.read_csv("clean_test.csv",usecols=['utterance', 'intent'])
    sent_csv=pd.read_excel("hiri_data_25030.xlsx",sheet_name="Test",usecols=['utterance', 'intent1','intent'])
    sentence=sent_csv.utterance.values
    # sentence=[sentence[0]]

    # data=pd.read_excel("RNN - Data1.xlsx",sheet_name="Sheet1")
    dmap=pd.read_excel("hiri_data_25030.xlsx",sheet_name="Mapping")

    # d= dict([(i,a) for i, a in zip(dmap.Intent, dmap.mapping)])
    # print(d)

    d= dict([(i,a) for i, a in zip(dmap.mapping, dmap.intent)])
    print(d)
    maxindx,predictions,maxindx1,maxindx2,maxindx3,maxindx4,predictions1,predictions2,predictions3,predictions4=[],[],[],[],[],[],[],[],[],[]
    top_p1,top_p2,top_p3,top_class1,top_class2,top_class3=[],[],[],[],[],[]
    for index,value in enumerate(sentence):
        print(value)
        review = str(value)
        review = " ".join(review.split())
        inputs = tokenizer.encode_plus(
                review, None, add_special_tokens=True, max_length=max_len
            )
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]
        padding_length = max_len - len(ids)
        ids = ids + ([0] * padding_length)
        mask = mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)
        ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
        mask = torch.tensor(mask, dtype=torch.long).unsqueeze(0)
        token_type_ids = torch.tensor(token_type_ids, dtype=torch.long).unsqueeze(0)
        ids = ids.to(DEVICE, dtype=torch.long)
        token_type_ids = token_type_ids.to(DEVICE, dtype=torch.long)
        mask = mask.to(DEVICE, dtype=torch.long)
        outputs = MODEL(ids=ids, mask=mask, token_type_ids=token_type_ids)
        maxidx=torch.argmax(outputs[0],keepdim=False)
        maxindx.append(int(maxidx))
        print(int(maxidx))
        
        # for intent, cat in d.items():
            # if cat == int(maxidx):
                # print(intent)
        
        print(d[int(maxidx)])  
        predictions.append(d[int(maxidx)])
        # outputs = torch.sigmoid(outputs).cpu().detach().numpy()
        print(str(outputs[0]))
        
        prob = nnf.softmax(outputs[0], dim=1)
        top_p, top_class = prob.topk(3, dim = 1)
        print("confidence score",top_p)
        print("confidence score1",top_p[0][0])
        
        top_p1.append(float(top_p[0][0]))
        top_p2.append(float(top_p[0][1]))
        top_p3.append(float(top_p[0][2]))
        print("top_class",top_class)
        print("top_class1",top_class[0][0])
        
        top_class1.append(int(top_class[0][0]))
        top_class2.append(int(top_class[0][1]))
        top_class3.append(int(top_class[0][2]))
        
        predictions1.append(d[int(top_class[0][0])])
        predictions2.append(d[int(top_class[0][1])])
        predictions3.append(d[int(top_class[0][2])])
        
    sent_csv["pred_idx"]=pd.DataFrame(maxindx)
    sent_csv["pred_values"]=pd.DataFrame(predictions)
    sent_csv["top_p1"]=pd.DataFrame(top_p1)
    sent_csv["top_p2"]=pd.DataFrame(top_p2)
    sent_csv["top_p3"]=pd.DataFrame(top_p3)
    sent_csv["top_class1"]=pd.DataFrame(top_class1)
    sent_csv["top_class2"]=pd.DataFrame(top_class2)
    sent_csv["top_class3"]=pd.DataFrame(top_class3)
    sent_csv["predictions1"]=pd.DataFrame(predictions1)
    sent_csv["predictions2"]=pd.DataFrame(predictions2)
    sent_csv["predictions3"]=pd.DataFrame(predictions3)

    sent_csv.to_csv("new_iris_sentcsv"+str(i)+".csv")

# In[ ]: