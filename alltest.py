"""
朴素贝叶斯公式:
P(theta|sen)=P(sen|theta)P(theta)/P(sen)
P(sen):在句子已给定时可看做常数，所以可以不计算，
P(sen|theta)＝P(w1|theta)P(w2|theta)... P(wn, theta)
P(theta):情感值为０或１的概率
"""
from collections import defaultdict
from math import log
import pandas as pd
from sklearn.model_selection import train_test_split
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk import word_tokenize
from numpy import nonzero


def preprocess_review(review):
    bs_process = BeautifulSoup(review, builder="lxml")
    stop_words = set(stopwords.words('english'))
    the_text = bs_process.get_text().tolower()
    words = word_tokenize(the_text)
    words = [w for w in words if w not in stop_words]
    return words


train_data_review = []
test_data_review = []
file_name = '/home/hanzhao/CSV_FILES/labeledTrainData.tsv'
all_data = pd.read_csv(file_name, header=0, sep='\t', quoting=3)
train_data, test_data = train_test_split(all_data, test_size=0.2)
for every_piece in train_data["review"]:
    train_data_review.append(preprocess_review(every_piece))
for every_piece in test_data["review"]:
    test_data_review.append(preprocess_review(every_piece))
the_positive_num = len(nonzero(train_data['sentiment'])[0])
the_negative_num = len(train_data['sentiment'])-the_positive_num
the_categoried_vocabulary = defaultdict(dict)
the_train_sen_rev_pair = zip(train_data_review, train_data['sentiment'])
for every_review, sentiment in the_train_sen_rev_pair:
    for every_word in every_review:
        the_categoried_vocabulary[sentiment][every_word] += 1
the_test_sen_rev_pair = zip(test_data_review, test_data['sentiment'])

word_num_every_category = dict()
voca_size_every_category = dict()

for every_tag in the_categoried_vocabulary:
    voca_size_every_category[every_tag] = len(the_categoried_vocabulary[every_tag])
    word_num_every_category[every_tag] = sum(the_categoried_vocabulary[every_tag].values())

for every_review, sentiment in the_test_sen_rev_pair:
    sen_prob_pair = defaultdict()
    for every_sentiment in the_categoried_vocabulary:
        words_num = word_num_every_category[sentiment]
        voca_size = voca_size_every_category[sentiment]
        for word in every_review:
            word_num = the_categoried_vocabulary[sentiment][word]
            sen_prob_pair[every_sentiment] += log((word_num+1)/(words_num+voca_size))
        result = min(sen_prob_pair)




