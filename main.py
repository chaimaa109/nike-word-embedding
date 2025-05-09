# Nike Word Embedding Project

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
import numpy as np

# load data
try:
    df = pd.read_csv("NikeProductDescriptions.csv")
except:
    print("Make sure the NikeProductDescriptions.csv file is in this folder.")
    exit()

# filter subtitles
valid_subs = ["Men's Shoes", "Men's T-Shirt", "Women's Shoes", "Skate Shoes", "Older Kids' T-Shirt", "Shorts"]
df['subtitle'] = df['subtitle'].fillna('')
df['subtitle'] = df['subtitle'].apply(lambda x: 'Shorts' if 'Shorts' in x else x)
df = df[df['subtitle'].isin(valid_subs)]

# descriptions
texts = df['description'].fillna('').tolist()
labels = df['subtitle'].tolist()

# plotting function
def plot_embedding(X, title):
    plt.figure(figsize=(10, 7))
    colors = {
        "Men's Shoes": 'red',
        "Women's Shoes": 'blue',
        "Men's T-Shirt": 'green',
        "Older Kids' T-Shirt": 'purple',
        "Skate Shoes": 'orange',
        "Shorts": 'cyan'
    }
    for label in set(labels):
        idxs = [i for i, l in enumerate(labels) if l == label]
        plt.scatter(X[idxs, 0], X[idxs, 1], label=label, alpha=0.5, color=colors.get(label, 'gray'))
    plt.legend()
    plt.title(title)
    plt.grid(True)
    plt.show()

# BOW
vec_bow = CountVectorizer(stop_words='english')
X_bow = vec_bow.fit_transform(texts).toarray()
bow_2d = PCA(n_components=2).fit_transform(X_bow)
plot_embedding(bow_2d, "Bag of Words")

# TF-IDF
vec_tfidf = TfidfVectorizer(stop_words='english')
X_tfidf = vec_tfidf.fit_transform(texts).toarray()
tfidf_2d = PCA(n_components=2).fit_transform(X_tfidf)
plot_embedding(tfidf_2d, "TF-IDF")

# Word2Vec
sentences = [t.lower().split() for t in texts]
w2v_model = Word2Vec(sentences=sentences, vector_size=100, window=5, min_count=1, workers=4, seed=42)

def vec_from_sent(sent, model):
    words = sent.lower().split()
    vecs = [model.wv[w] for w in words if w in model.wv]
    return np.mean(vecs, axis=0) if vecs else np.zeros(model.vector_size)

X_w2v = np.array([vec_from_sent(t, w2v_model) for t in texts])
w2v_2d = PCA(n_components=2).fit_transform(X_w2v)
plot_embedding(w2v_2d, "Word2Vec")
