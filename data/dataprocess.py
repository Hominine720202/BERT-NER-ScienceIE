import pandas as pd
import numpy as np
import os,sys
from pytorch_pretrained_bert import BertTokenizer
import stanfordnlp
import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
nlp = stanfordnlp.Pipeline() 
tokenizer = BertTokenizer.from_pretrained("../scibert/", do_lower_case=True)
"""
def sent_tokenize(sentence):
    doc = nlp(sentence)
    result = []
    for i, sentence in enumerate(doc.sentences):
        sent = ' '.join(word.text for word in sentence.words)
        result.append(sent)
    del doc
    return result
"""
def word_tokenize(sentence):
    return tokenizer.tokenize(sentence)
# 使用 bert 的 tokenizer
# 使用贪心算法，从区间集合中去除最少的区间以使得整体不重叠
def eraseOverlapIntervals(intervals):
    if not len(intervals):
        return 0
    result = [intervals[0]]
    #按照end进行排序
    intervals = sorted(intervals, key=lambda k: k[1])
    #记录最小的end
    minEnd = intervals[0][2]
    rest = 1
    for i in range(1,len(intervals)):
        #若下一个interval的start小于minEnd,则算重叠
        if intervals[i][1] < minEnd:
            continue
        result.append(intervals[i])
        rest += 1
        minEnd = intervals[i][2]
    res = []
    for item in result:
        if item not in res:
            res.append(item)
    return res
# 给定 sentence，给定标注范围与类型，获得标注序列

# 给定 sentence，给定标注范围与类型，获得标注序列

def BIOextract(sentence,ann,start=0):
    words = word_tokenize(sentence)
    selectedWord = word_tokenize(sentence[ann[1]-start:ann[2]-start])
    label = ['O' for idx in range(len(words))]
    ite,rte = 0,0
    lenSelect = len(selectedWord)
    for idx in range(len(words)):
        tempWords = words[idx:idx+len(selectedWord)]
        if tempWords == selectedWord:
            if len(selectedWord) == 1:
                label[idx] = 'S-' + ann[0]
            else:
                label[idx] = 'B-' + ann[0]
                for lIdx in range(idx+1,idx+lenSelect-1):
                    label[lIdx] = 'I-' + ann[0]
                label[idx+lenSelect-1] = 'E-' + ann[0]
    """
    for w in words:
        rte = 0
        for t in selectedWord:
            if w == t:
                if rte == 0:
                    label[ite] = 'B-' + ann[0]
                else:
                    label[ite] = 'I-' + ann[0]
                continue
            rte = rte + 1
        ite = ite + 1
    """
    return label
# 同一个语句的两个标注序列进行合并
def combineLabel(labela,labelb):
    for idx in range(len(labela)):
        if labela[idx] == 'O' and labelb[idx] != 'O':
            labela[idx] = labelb[idx]
    return labela
# 处理一个摘要文件
def abs2sent(idx,folder):
    absTxt = open(folder+"/"+idx+'.txt', 'r')
    abstract = absTxt.read()
    sentences = sent_tokenize(abstract)
    annFile = open(folder+"/"+idx+'.ann', 'r')
    annLines = annFile.readlines()
    anns = []
    label = []
    sentLen = [] # 分句的长度
    producedSentences = []
    # 获取文段中每一个sentence的区间
    for sentence in sentences:
        if len(sentLen) != 0:
            sentLen.append([sentLen[-1][1]+1,sentLen[-1][1]+1+len(sentence)])
        else:
            sentLen.append([0,len(sentence)])
    # 处理标注格式
    for line in annLines:
        lineSpl = line.split("\t")
        if lineSpl[0].find("T") != -1:
            ann = lineSpl[1].split(" ")
            try:
                ann[1] = int(ann[1])
                ann[2] = int(ann[2])
            except:
                print(idx,line)
                break
            anns.append(ann)
    # 获得不重叠区间
    anns = eraseOverlapIntervals(anns)
    labels = []
    # 进行标注 + 合并标注
    out = False
    for sentIdx in range(len(sentences)):
        tmpAnn = None
        if sentIdx != 0:
            start = start + len(sentences[sentIdx-1])+1
        else:
            start = 0
        for ann in anns:
            if ann[1] >= sentLen[sentIdx][0] and ann[2] <= sentLen[sentIdx][1]:
                if tmpAnn == None:
                    tmpAnn = BIOextract(
                        sentences[sentIdx],
                        ann,
                        start = start
                    )
                else:
                    tmpAnn = combineLabel(
                        BIOextract(
                            sentences[sentIdx],
                            ann,
                            start = start
                        ),
                        tmpAnn
                    )
        if tmpAnn == None:
            tmpAnn = ['O' for idx in range(len(word_tokenize(sentences[sentIdx])))]
        labels.append(tmpAnn)
        producedSentences.append(sentences[sentIdx])
    return labels,producedSentences
# 处理一组摘要文件
def tF(arr,folder):
    labels = []
    sentences = []
    for item in arr:
        tmpLabel,tmpSentence = abs2sent(item,folder)
        labels = labels + (tmpLabel)
        sentences = sentences + (tmpSentence)
    return labels,sentences
# 写入tags.txt
def writeLabel(arr,filename):
    l=arr
    f=open(filename,"w",encoding="utf-8")
    for line in l:
        for chac in line:
            f.write(chac+" ")
        f.write("\n")
    f.close()
# 写入sentence.txt
def writeTxt(arr,filename):
    l=arr
    f=open(filename,"w",encoding="utf-8")
    for line in l:
        f.write(line+"\n")
    f.close()
import random
def writeData(labels,txts,name):
    writeLabel(labels,"selfdata/"+name+"/tags.txt")
    writeTxt(txts,"selfdata/"+name+"/sentences.txt")
# 对所有的文件进行处理
def traveFiles(folder):
    result = []
    filenames = os.listdir(folder)
    for item in filenames:
        if item.find('.txt') != -1:
            result.append(item[:-4])
    random.shuffle(result)
    labels,txt = tF(result,folder)
    writeData(labels,txt,folder)
    print(len(labels))
def makeFile(folder):
    # files 存储所有的样例文件名
    files = os.listdir(folder)
    anns,txts = [],[]
    for item in files:
        if item.find('.ann') != -1:
            anns.append(item)
            txts.append(item[:-4]+".txt")
    traveFiles(folder)
makeFile("train")
makeFile("val")
makeFile("test")
