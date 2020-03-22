# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import numpy as np
import os,sys
import nltk
from nltk.tokenize import sent_tokenize
from pytorch_pretrained_bert import BertTokenizer


# %%
tokenizer = BertTokenizer.from_pretrained('../scibert', do_lower_case=True)


# %%
def splitSentence(abstract):
    return sent_tokenize(abstract)


# %%
def word_tokenize(sentence):
    return tokenizer.tokenize(sentence)
# ä½¿ç”¨ bert çš„ tokenizer


# %%
files = os.listdir('./exter/')


# %%
sentences = []
for file in files:
    if file.find('.txt') != -1:
        absTxt = open('./exter/'+file, 'r',encoding="utf-8")
        abstract = absTxt.read()
        for s in sent_tokenize(abstract):
            sentences.append(s)


# %%
f=open("./exter_train/train/sentences.txt","w",encoding="utf-8")
for line in sentences:
    f.write(line+"\n")
f.close()


# %%
f=open("./exter_train/train/tags.txt","w",encoding="utf-8")
for line in sentences:
    tags = ['O' for idx in word_tokenize(line)]
    for t in tags:
        f.write(t+" ")
    f.write("\n")
f.close()


# %%


