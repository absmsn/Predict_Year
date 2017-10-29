"""
朴素贝叶斯公式:
P(theta|sen)=P(sen|theta)P(theta)/P(sen)
P(sen):在句子已给定时可看做常数，所以可以不计算，
P(sen|theta)＝P(w1|theta)P(w2|theta)... P(wn, theta)
P(theta):情感值为０或１的概率
"""
import re
import logging
from collections import defaultdict
from math import log
import pandas as pd
from sklearn.model_selection import train_test_split
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk import word_tokenize
from numpy import nonzero
from sklearn.metrics import accuracy_score


def preprocess_review(review):
    bs_process = BeautifulSoup(review, "lxml")
    stop_words = set(stopwords.words('english'))
    text = bs_process.get_text()
    text = re.sub('[^\d\w]', ' ', text).lower()
    words = word_tokenize(text)
    words = [w for w in words if w not in stop_words]
    return words

log_file_name = '/home/hanzhao/nb.log'
logging.basicConfig(level=logging.DEBUG, format='%(message)s', filename=log_file_name, filemode='w')


train_data_review = [] 
test_data_review = []
file_name = '/home/hanzhao/CSV_FILES/labeledTrainData.tsv'
all_data = pd.read_csv(file_name, header=0, sep='\t', quoting=3)
train_data, test_data = train_test_split(all_data, test_size = 0.2)
for every_piece in train_data["review"]:
    train_data_review.append(preprocess_review(every_piece))
for every_piece in test_data["review"]:
    test_data_review.append(preprocess_review(every_piece))

sentiment_prob = defaultdict(int)
sentiment_prob[1] = len(nonzero(train_data['sentiment'])[0])
sentiment_prob[0] = len(train_data['sentiment'])-sentiment_prob[1]

categoried_vocabulary = defaultdict(dict)
train_sen_rev_pair = zip(train_data_review, train_data['sentiment'])
for every_review, sentiment in train_sen_rev_pair:
    for every_word in every_review:
        if every_word not in categoried_vocabulary[sentiment]:
            categoried_vocabulary[sentiment][every_word] = 1
        else:
            categoried_vocabulary[sentiment][every_word] += 1
test_sen_rev_pair = zip(test_data_review, test_data['sentiment'])

word_num_every_category = dict()
voca_size_every_category = dict()

for every_tag in categoried_vocabulary:
    voca_size_every_category[every_tag] = len(categoried_vocabulary[every_tag])
    word_num_every_category[every_tag] = sum(categoried_vocabulary[every_tag].values())

result_set = []

for every_review, sentiment in test_sen_rev_pair:
    sen_prob_pair = defaultdict(int)
    for every_sentiment in categoried_vocabulary: 
        words_num = word_num_every_category[every_sentiment]
        voca_size = voca_size_every_category[every_sentiment]
        for word in every_review:
            if word in categoried_vocabulary[every_sentiment]:
                word_num = categoried_vocabulary[every_sentiment][word]
            else:
                word_num = 0
            sen_prob_pair[every_sentiment] += log((word_num+1)/(words_num+voca_size))
    _sen_prob_pair = {v:k for k,v in sen_prob_pair.items()} 
    result = _sen_prob_pair[max(_sen_prob_pair)]  
    result_set.append(result)


print(accuracy_score(test_data['sentiment'],result_set))