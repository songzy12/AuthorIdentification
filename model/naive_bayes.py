import numpy as np

from sklearn import naive_bayes
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import model_selection, metrics


def run_mnb(train_tfidf, test_tfidf, train_y):
    cv_scores = []
    pred_full_test = 0
    pred_train = np.zeros([train_tfidf.shape[0], 3])

    kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2017)
    for dev_index, val_index in kf.split(train_tfidf):
        dev_X, val_X = train_tfidf[dev_index], train_tfidf[val_index]
        dev_y, val_y = train_y[dev_index], train_y[val_index]

        model = naive_bayes.MultinomialNB()
        model.fit(dev_X, dev_y)
        pred_val_y = model.predict_proba(val_X)
        pred_test_y = model.predict_proba(test_tfidf)

        pred_full_test = pred_full_test + pred_test_y
        pred_train[val_index, :] = pred_val_y
        cv_scores.append(metrics.log_loss(val_y, pred_val_y))
    print("Mean cv score : ", np.mean(cv_scores))
    pred_full_test = pred_full_test / 5.0
    return pred_train, pred_full_test


def run_mnbs(train_df, test_df, train_y):
    # Fit transform the count vectorizer.
    tfidf_vec = CountVectorizer(ngram_range=(1, 2))
    tfidf_vec.fit(train_df["text"].values.tolist() +
                  test_df["text"].values.tolist())
    train_tfidf = tfidf_vec.transform(train_df["text"].values.tolist())
    test_tfidf = tfidf_vec.transform(test_df["text"].values.tolist())

    pred_train, pred_full_test = run_mnb(train_tfidf, test_tfidf, train_y)

    # Add the predictions as new features.
    train_df["nb_cvec_eap"] = pred_train[:, 0]
    train_df["nb_cvec_hpl"] = pred_train[:, 1]
    train_df["nb_cvec_mws"] = pred_train[:, 2]
    test_df["nb_cvec_eap"] = pred_full_test[:, 0]
    test_df["nb_cvec_hpl"] = pred_full_test[:, 1]
    test_df["nb_cvec_mws"] = pred_full_test[:, 2]
    print("Naive Bayesian Count Vector finished...")

    # Fit transform the tfidf vectorizer.
    tfidf_vec = TfidfVectorizer(ngram_range=(1, 2))
    tfidf_vec = tfidf_vec.fit(train_df["text"].values.tolist() +
                              test_df["text"].values.tolist())
    train_tfidf = tfidf_vec.transform(train_df["text"].values.tolist())
    test_tfidf = tfidf_vec.transform(test_df["text"].values.tolist())

    pred_train, pred_full_test = run_mnb(train_tfidf, test_tfidf, train_y)

    # Add the predictions as new features.
    train_df["nb_tfidf_eap"] = pred_train[:, 0]
    train_df["nb_tfidf_hpl"] = pred_train[:, 1]
    train_df["nb_tfidf_mws"] = pred_train[:, 2]
    test_df["nb_tfidf_eap"] = pred_full_test[:, 0]
    test_df["nb_tfidf_hpl"] = pred_full_test[:, 1]
    test_df["nb_tfidf_mws"] = pred_full_test[:, 2]
    print("Naive Bayersian TFIDF Vector finished...")
    return train_df, test_df
