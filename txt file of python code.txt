# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 17:54:00 2019

@author: Amul
"""


import glob
import string
import os
import nltk
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from yellowbrick.text import FreqDistVisualizer
from wordcloud import WordCloud
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.cluster.hierarchy import ward, dendrogram


file = glob.glob(os.path.join(os.getcwd(), "texts", "*.txt"))
datas = []
for text in file:
    with open(text) as f:
        txt = f.read() 
        datas.append(txt)

def process_word(text, stem = True) :
    Table = str.maketrans(dict.fromkeys(string.punctuation))
    Text = text.translate(Table)
    tokens = word_tokenize(Text)
    tokens= [word for word in tokens if word.isalpha()]
    stopwords = nltk.corpus.stopwords.words('english')
    tokens = [w for w in tokens if not w in stopwords]
    if stem:
        stemmer = SnowballStemmer("english")
        tokens = [stemmer.stem(word) for word in tokens]
    return tokens


tfidf_vectorizer = TfidfVectorizer( tokenizer= process_word,
                                    max_df=0.4,
                                    min_df=0.1, stop_words='english',
                                    use_idf=True)
vectorizer = CountVectorizer(tokenizer=process_word,
                            max_df = 0.4,
                            min_df = 0.1, 
                            stop_words = 'english')
tfidf_matrix = tfidf_vectorizer.fit_transform(datas)

tf_model = vectorizer.fit_transform(datas)
total_tf = [sum(x) for x in zip(*tf_model.toarray())]
terms = tfidf_vectorizer.get_feature_names()
term_list = []
popTerm = []
pt = []
freq = []
cloud_word = {}
for i,v in enumerate(total_tf):
    if v > 25:
        freq.append(v)
        popTerm.append(i)
        term_list.append(terms[i])
        cloud_word[terms[i]] = v
        pt.append(tf_model.transpose().toarray()[i])
print(freq)
print(term_list)

visualizer = FreqDistVisualizer(features=terms, orient='v')
visualizer.fit(tf_model)
visualizer.show()

wordcloud = WordCloud(normalize_plurals= False).generate_from_frequencies(cloud_word)    

plt.imshow(wordcloud)
plt.axis("off")
plt.show()

km = KMeans(n_clusters= 2)
km = km.fit(tf_model.transpose())
print(tf_model.shape)
clusters = km.labels_.tolist()

color = [ '#0356fc' if x == 0 else '#fc0341' for x in clusters]
figure, a = plt.subplots(figsize=(25,20))


for i in popTerm:
    a.scatter(tf_model.toarray()[2][i], tf_model.toarray()[3][i], c =color[i])
    a.annotate(terms[i], (tf_model.toarray()[2][i], tf_model.toarray()[3][i]))
    print(terms[i])
plt.show()

distance_matrix = euclidean_distances(pt)
link = ward(distance_matrix)
figure, a = plt.subplots(figsize=(25,20))
a = dendrogram(link, labels= term_list, orientation='top')
plt.tick_params(\
    which='both',
    bottom='off',
    top='off',
    axis= 'x',
    labelbottom='off')
plt.tight_layout()
plt.show()

