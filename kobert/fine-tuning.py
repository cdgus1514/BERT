# KB

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from pytorch_pretrained_bert import BertTokenizer, BertConfig
from pytorch_pretrained_bert import BertAdam, BertForSequenceClassification
from tqdm import tqdm, trange
import pandas as pd
import io
import numpy as np



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
print(torch.cuda.get_device_name(0), end="\n\n")



# df = pd.read_csv("kobert/data/in_domain_train.tsv", delimiter="\t", header=None, names=["sentence_source", "label", "label_notes", "sentence"])
# print(df.shape, end="\n\n\n")
# print(df.sample(10), end="\n\n\n")

# sentences = df.sentence.values

df = pd.read_csv("kobert/data/train.csv", delimiter=",")
print(df.shape, end="\n\n\n")
# (296463, 4)

print(df.sample(10), end="\n\n\n")
'''
            id year_month                                               text  smishing
213549  242671    2018-03  XXX 고객님오늘 업무처리해드렸던 XXX 민락동 XXX대리입니다. 꾸준하게 저희 X...         0
231496  262953    2018-05  XXX 고객님안녕하세요 지난주 금요일 업무도와드린 XXX은행 XXX점 XXX대리입니...         0
82255    96013    2017-05  XXX 고객님오늘 업무처리 도와드린 XXX은행 옥련동지점 XXX 과장입니다혹시 거래...         0
272990  309802    2018-10            매우동의하시도록XXX영등포지점직원모두는최선을다하겠습니다.행복한저녁되세요         0
272986  309798    2018-10  XXX 고객님!아침저녁 느껴지는 신선함에 위대한 자연의 힘의 느껴지는 월요일 시작입...         0
21445    25109    2017-03                 꾸준히 거래해 주셔서 대단히 감사합니다XXX은행강변역 XXX.         0
251265  285214    2018-07  (광고) 연금이체  혜택 안내XXX 고객님! 소중한 연금 받으실 때는 XXX 골든라...         0
138685  159502    2017-09              96(수)적금이 만기입니다.시간내서 내점바랍니다.XXX은행 탄현지점         0
259122  293924    2018-08                무더웠던여름이이제드디어지나가나봅니다 즐거운 주말되세요 XXX팀장         0
32309    37802    2017-04                 내점해주셔서 감사와 매우동의부탁드림니다XXX은행수유동XXX올림         0
'''


sentences = df.text.values


## KoBERT tokenizer import
from gluonnlp.data import SentencepieceTokenizer
from kobert.utils import get_tokenizer

tok_path = get_tokenizer()
sp = SentencepieceTokenizer(tok_path)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)



## KoBERT Tokenizer
tokenized_texts = [sp(text) for text in sentences]
print(tokenized_texts[0])
# # ['▁', 'X', 'X', 'X', '은행', '성', '산', 'X', 'X', 'X', '팀장', '입니다', '.', '행복', '한', '주', '말', '되', '세요']



# 문장 최대길이
MAX_LEN = 128

# 문장의 토큰들 인덱스처리
# input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
# for x in tokenized_texts:
#     if x == '▁':
#         pass
#     else:
#         input_ids = [tokenizer.convert_tokens_to_ids(x)]






def tokenize(sentence):
    stop_ward = ['X', 'XXX', '.', '을', '를', '이', '가', '-', '(', ')', ':', '!', '?', ')-', '.-', '▁', 'ㅡ', 'XXXXXX', '..', '.(', '은', '는' ]
    word_bag = []

    for word in sentence:
        
        if word in stop_ward:
            continue
        else:
            word_bag.append(word)
    
    result = ' '.join(word_bag)

    return result


result = [tokenize(sentence) for sentence in tokenized_texts]

print(result[0])