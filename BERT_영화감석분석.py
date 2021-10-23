# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import urllib.request
import pandas as pd
import torch

# data 불러오기 => 네이버 평점에서 가져온 데이터
urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt", filename="ratings_train.txt")
train_data = pd.read_table('ratings_train.txt')
train_data['document']

train_data

# +
import pandas as pd
from transformers import BertTokenizer

# BERT의 경우 문장 처음/끝 부분에 각각 표시를 해두어야 하나의 문장으로 생각하고 탐색할 수 있다
Pretreatment_train_data = train_data['document'].str.replace(pat=r'[^\w]', repl=r'', regex=True) # 특수문자 제거
sentences = ["[CLS]" + str(s) + "[SEP]" for s in Pretreatment_train_data] # BERT 양식에 맞춰 문장 앞뒤에 붙임
# Bert Token의 경우 Word Piece Tokenizing으로 BPT라고도 불린다.
# Word Piece Tokenizing => OOV(out of Vocabulary)를 없애기 위해 기존 Word2Vec과는 다르게 사용되어진다. (자세한 내용 참조..)

# BertTokenizer.from_pretrained은 Bert(해당케이스)에서 sentence를 자동으로 Word Piece Tokenizing 해주는 기법이다.
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased", do_lower_case = False)
tokenizer_text = [tokenizer.tokenize(s) for s in sentences]
# -

print(tokenizer_text[0])


# +
# 문장의 최대 길이를 구하는 이유는 최대길이를 기준으로 Zero-padding(Vector 차원을 동일하게 해줌)을 실시할 수 있기 때문임
def get_max_length(df):
    max_length = 0
    for row in df:
        if len(row) > max_length:
            max_length = len(row)
    return max_length

max_length = get_max_length(tokenizer_text) # 문장의 길이를 맞추기 위해서 max_length를 구하는 방법 => Zero_Padding
print(max_length)
# -

# tokenizer.convert_tokens_to_ids(x)는 중복을 최대한 제거한 후 단어 : 숫자 형식으로 변환시키는 기법이다
input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenizer_text] # 중복을 제거한 최대 단어 수를 확인 후 숫자로 변경
print(input_ids[0:9])

# +
from keras.preprocessing.sequence import pad_sequences

# pad_sequences는 max_length에 맞춰 최대 길이로 맞춰 넣는 것을 의미한다.(Zero-padding)
input_ids = pad_sequences(input_ids, maxlen = max_length, dtype = 'long', truncating = "post", padding = "post")

print(input_ids)


# +
# Attention_Mask는 의미있는 단어는 1, 의미없는 단어(Zero-padding)는 0으로 둠으로써 계산식이 많아지지 않도록 조절해주는 역할을 한다. 

Mask = []

for seq in input_ids:
    mask = [float(i>0) for i in seq]
    Mask.append(mask)

# +
# numpy를 안쓰고 torch를 쓰는 이유 :
# input_ids 는 2D 텐서이고 크기는 max_len * len(input_ids)로 추정됨
train_inputs = torch.tensor(input_ids)
train_labels = torch.tensor(train_data['label'])
train_masks = torch.tensor(Mask)

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
batch_size = 32
train_tensordata = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_tensordata) # train_data를 무작위로 섞는다?
train_dataloader = DataLoader(train_tensordata, sampler=train_sampler, batch_size=batch_size) # train_sampler를 batch_size만큼 나눈다.
print(len(train_dataloader))
print(train_dataloader)
print(150000/32)
# -

# pytorch와 cuda가 서로 연동되지 않는다 -> 버젼의 문제라고 판단되며 해결방안이 필요하다
# Cuda는 컴퓨터 내에 내장된 GPU를 사용하기 위한 프레임워크이며 내 PC는 현재 GPU가 없다.. => 훈련속도 매우 늦을 것으로 판단됨
# !pip install torch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0

# +
import torch

if torch.cuda.is_available():    
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print('No GPU available, using the CPU instead.')


# +
from transformers import BertForSequenceClassification, AdamW, BertConfig
# BertForSequenceClassification는 BERT 모델을 분류분석할 때 사용되는 기능임
model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=2)

# GPU가 있다면 model.cuda() , CPU로 할 경우 model.cpu()로 진행하면 된다.
# 훈련 결과 CPU만으로는 학습이 절대 불가능하다..
model.cpu()
# -

from transformers import get_linear_schedule_with_warmup
# AdamW => Adam알고리즘 중 가중치 감쇠 수정한 것, lr(Learning rate) : 학습률, eps => 수치적 안전성을 위한 앱실론...?, epoch => 몇회를 학습시킬 것인가
# total_steps => batch_size * epochs 를 의미한다.
# scheduler => 
optimizer = AdamW(model.parameters(), lr = 2e-5, eps = 1e-8)
epochs = 4
total_steps = len(train_dataloader) * epochs
# 스케쥴러는 알고리즘이 학습할 때 일정한 학습률로 학습할 수 있도록 일정을 조정하는 역할을 함 => 시간표/다이어리와 같다고 생각하면 될듯
scheduler = get_linear_schedule_with_warmup(optimizer,
                                           num_warmup_steps = 0,
                                           num_training_steps = total_steps)

# +
import datetime
import time
# 정확도를 확인하는 지표 : 맞힌것 / 전체 개수를 확인하는 것으로 지표 기능이 따로 있는 것으로 알고 있는데....?
def flat_accruracy(preds, labels):
    
    pred_flat = np.argmax(preds, axis= 1).flatten()
    lables_flat = labels.flatten()
    
    return np.sum(pred_flat = labels_flat) / len(labels_flat)

# 학습시간이 얼마나 걸리는지를 확인하는 것
def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    
    return str(datetime.timedelta(seconds=elapsed_rounded))


# -

import random
import numpy as np
seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)

model.zero_grad()

# +

for epoch_i in range(0, epochs):
    print("")
    print("========== Epoch {:}/{:}".format(epoch_i + 1, epochs))
    t0 = time.time()
    total_loss = 0
    model.train()
    
    for step, batch in enumerate(train_dataloader):
        if step % 500 == 0 and not step == 0 :
            elapsed = format_time(time.time() - t0)
            print( " Batch {:>5,} of {:>5,}. Elapsed: {:}".format(step, len(train_dataloader), elapsed) )
        
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        
        output = model(b_input_ids,
                      token_type_ids = None,
                      attention_mask = b_input_mask,
                      labels = b_labels)
        loss = output[0]
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        optimizer.step()
        scheduler.step()
        model.zero_grad()
        
    
# -


