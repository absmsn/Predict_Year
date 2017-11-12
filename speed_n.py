import re
from collections import Counter
import pandas as pd
from numpy import zeros
from numpy import array
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from nltk import word_tokenize



def review_to_words(raw_review):
    review_text = BeautifulSoup(raw_review, 'lxml').get_text()
    letter_only = re.sub('[^a-zA-Z]', ' ', review_text)
    words = letter_only.lower().split()
    stop_words = set(stopwords.words('english'))
    meaningful_words = [w for w in words if w not in stop_words]
    return meaningful_words


def review_to_string(raw_review):
    meaningful_words = review_to_words(raw_review)
    return ''.join(meaningful_words)


train_data_review = []
test_data_review = []
train_file_name = '/home/hanzhao/CSV_FILES/labeledTrainData.tsv'
all_data = pd.read_csv(train_file_name, header=0, sep='\t', quoting=3)
train_data, test_data = train_test_split(all_data, test_size=0.2)
num_train_review = train_data["review"].shape[0]
num_test_review = test_data["review"].shape[0]
for every_data in train_data["review"]:
    train_data_review.append(review_to_string(every_data))
for every_data in test_data["review"]:
    test_data_review.append(review_to_string(every_data))
vectorizer = CountVectorizer(analyzer="word")
train_data_features = vectorizer.fit_transform(train_data_review)
test_data_features = vectorizer.transform(test_data_review)

