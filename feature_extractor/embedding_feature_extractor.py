import numpy as np
from tqdm import tqdm
import nltk


# load the GloVe vectors in a dictionary:
def load_word_vecs():
    embeddings_index = {}
    wv = "./input/glove.6B.100d.txt"
    f = open(wv)
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype="float32")
        embeddings_index[word] = coefs
    f.close()
    print("Found %s word vectors." % len(embeddings_index))
    return embeddings_index


def sent2vec(embeddings_index, s):
    # this function creates a normalized vector for the whole sentence
    words = str(s).lower()
    words = nltk.word_tokenize(words)
    words = [w for w in words if not w in nltk.corpus.stopwords.words("english")]
    words = [w for w in words if w.isalpha()]
    M = []
    for w in words:
        try:
            M.append(embeddings_index[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    if type(v) != np.ndarray:
        return np.zeros(100)
    return v / np.sqrt((v ** 2).sum())


def add_glove_feature(x_train, x_test):
    embeddings_index = load_word_vecs()
    # create sentence vectors using the above function for training and validation set
    xtrain_glove = [sent2vec(embeddings_index, x) for x in tqdm(x_train)]
    xtest_glove = [sent2vec(embeddings_index, x) for x in tqdm(x_test)]
    xtrain_glove = np.array(xtrain_glove)
    xtest_glove = np.array(xtest_glove)
    return xtrain_glove, xtest_glove, embeddings_index
