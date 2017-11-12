from collections import Counter
from bs4 import BeautifulSoup
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

f_name = '/home/hanzhao/wikiextractor/enwiki/AB/Aoraki  Mount Cook'
symbols = set([',', '.', '(', ')', '/', '``', '\'\''])
stop_words = set(stopwords.words('english'))
with open(f_name, mode='r') as fi:
    data = fi.read()

data = BeautifulSoup(data)
data = data.get_text().lower()
words = word_tokenize(data)
words = [word for word in words if word not in stop_words]
cc = Counter(words)
print(cc)
