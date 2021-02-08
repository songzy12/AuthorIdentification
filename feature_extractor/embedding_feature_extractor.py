import numpy as np
import pandas as pd

from tqdm import tqdm
from nltk import word_tokenize
from nltk.corpus import stopwords


# load the GloVe vectors in a dictionary:
def load_word_vecs(wv="./input/glove.6B.100d.txt"):
    word_vecs = {}
    f = open(wv)
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype="float32")
        word_vecs[word] = coefs
    f.close()
    print("Found %s word vectors." % len(word_vecs))
    return word_vecs


def sent2vec(word_vecs, s):
    # this function creates a normalized vector for the whole sentence
    words = filter(lambda w: w not in stopwords.words("english")
                   and w.isalpha(), word_tokenize(str(s).lower()))
    M = []
    for w in words:
        if w in word_vecs:
            M.append(word_vecs[w])
    M = np.array(M)
    v = M.sum(axis=0)
    if type(v) != np.ndarray:
        return np.zeros(100)
    return v / np.sqrt((v ** 2).sum())


def add_glove_feature(train_df, test_df):
    word_vecs = load_word_vecs()
    # create sentence vectors using the above function for training and
    # test set
    train_df_glove = [sent2vec(word_vecs, x) for x in tqdm(train_df["text"])]
    test_df_glove = [sent2vec(word_vecs, x) for x in tqdm(test_df["text"])]
    train_df_glove = np.array(train_df_glove)
    test_df_glove = np.array(test_df_glove)

    train_df[["sent_vec_" + str(i) for i in range(100)]] = pd.DataFrame(
        train_df_glove.tolist()
    )
    test_df[["sent_vec_" + str(i) for i in range(100)]] = pd.DataFrame(
        test_df_glove.tolist()
    )

    return train_df, test_df, train_df_glove, test_df_glove, word_vecs
