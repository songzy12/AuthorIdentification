import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import LatentDirichletAllocation


def add_lda_feature(train_df, test_df, n_components=20):
    # analyzer : {'word', 'char', 'char_wb'}
    tfidf_vec = CountVectorizer(ngram_range=(1, 2))
    full_tfidf = tfidf_vec.fit_transform(
        train_df["text"].values.tolist() + test_df["text"].values.tolist()
    )
    train_tfidf = tfidf_vec.transform(train_df["text"].values.tolist())
    test_tfidf = tfidf_vec.transform(test_df["text"].values.tolist())

    lda = LatentDirichletAllocation(
        n_components=n_components,
        max_iter=5,
        learning_method="online",
        learning_offset=50.0,
        random_state=0,
    ).fit(full_tfidf)
    train_lda = pd.DataFrame(lda.transform(train_tfidf))
    test_lda = pd.DataFrame(lda.transform(test_tfidf))

    train_lda.columns = ["lda_" + str(i) for i in range(n_components)]
    test_lda.columns = ["lda_" + str(i) for i in range(n_components)]
    train_df = pd.concat([train_df, train_lda], axis=1)
    test_df = pd.concat([test_df, test_lda], axis=1)
    del full_tfidf, train_tfidf, test_tfidf, train_lda, test_lda

    print("LDA finished...")
    return train_df, test_df


def add_svd_feature(train_df, test_df, n_components=20):
    # Fit transform the tfidf vectorizer
    tfidf_vec = TfidfVectorizer(ngram_range=(1, 2))
    full_tfidf = tfidf_vec.fit_transform(
        train_df["text"].values.tolist() + test_df["text"].values.tolist()
    )
    train_tfidf = tfidf_vec.transform(train_df["text"].values.tolist())
    test_tfidf = tfidf_vec.transform(test_df["text"].values.tolist())

    svd_obj = TruncatedSVD(n_components=n_components, algorithm="arpack")
    svd_obj.fit(full_tfidf)
    train_svd = pd.DataFrame(svd_obj.transform(train_tfidf))
    test_svd = pd.DataFrame(svd_obj.transform(test_tfidf))

    train_svd.columns = ["svd_" + str(i) for i in range(n_components)]
    test_svd.columns = ["svd_" + str(i) for i in range(n_components)]
    train_df = pd.concat([train_df, train_svd], axis=1)
    test_df = pd.concat([test_df, test_svd], axis=1)
    del full_tfidf, train_tfidf, test_tfidf, train_svd, test_svd

    print("SVD finished...")
    return train_df, test_df
