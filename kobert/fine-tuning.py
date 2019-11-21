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
print(df.sample(10), end="\n\n\n")

sentences = df.text.values


## kobert tokenizer
from gluonnlp.data import SentencepieceTokenizer
from kobert.utils import get_tokenizer

tok_path = get_tokenizer()
sp = SentencepieceTokenizer(tok_path)


tokenized_texts = [sp(text) for text in sentences]
print(tokenized_texts[0])
print(tokenized_texts[10])
print(tokenized_texts[100])

MAX_LEN = 512

