import re
import pandas as pd
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from numpy import nonzero

def review_to_words(raw_review):
    review_text = BeautifulSoup(raw_review).get_text()
    letter_only = re.sub('[^a-zA-Z]', ' ', review_text)
    words = letter_only.lower().split()
    stop_words = set(stopwords.words('english'))
    meaningful_words = [w for w in words if w not in stop_words]
    return ''.join(meaningful_words)


train_data_review = []
test_data_review = []
train_file_name = '/home/hanzhao/CSV_FILES/labeledTrainData.tsv'
all_data = pd.read_csv(train_file_name, header=0, sep='\t', quoting=3)
train_data, test_data = train_test_split(all_data, test_set=0.3)
num_train_review = train_data["review"].shape[0]
num_test_review = test_data["review"].shape[0]
for i in range(num_train_review):
    train_data_review.append(review_to_words(train_data["review"][i]))
for j in range(num_test_review):
    test_data_review.append(review_to_words(test_data["review"][j]))
bb = BernoulliNB()
bb.fit(train_data_review, train_data["sentiment"])
result = bb.predict(test_data_review)
print((result-test_data["sentiment"])/num_test_review)