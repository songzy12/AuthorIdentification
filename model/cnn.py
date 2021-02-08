import numpy as np
from sklearn import model_selection

from keras.layers import Conv1D, MaxPooling1D, Flatten
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer


# NN
def doAddNN(X_train, X_test, pred_train, pred_test):
    X_train["nn_eap"] = pred_train[:, 0]
    X_train["nn_hpl"] = pred_train[:, 1]
    X_train["nn_mws"] = pred_train[:, 2]
    X_test["nn_eap"] = pred_test[:, 0]
    X_test["nn_hpl"] = pred_test[:, 1]
    X_test["nn_mws"] = pred_test[:, 2]
    return X_train, X_test


def initNN(nb_words_cnt, max_len):
    model = Sequential()
    model.add(Embedding(nb_words_cnt, 32, input_length=max_len))
    model.add(Dropout(0.3))
    model.add(Conv1D(64, 5, padding="valid", activation="relu"))
    model.add(Dropout(0.3))
    model.add(MaxPooling1D())
    model.add(Flatten())
    model.add(Dense(800, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation="softmax"))

    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    return model


def doNN(X_train, X_test, Y_train):
    max_len = 70
    nb_words = 10000

    print("Processing text dataset")
    texts_1 = []
    for text in X_train["text"]:
        texts_1.append(text)

    print("Found %s texts." % len(texts_1))
    test_texts_1 = []
    for text in X_test["text"]:
        test_texts_1.append(text)
    print("Found %s texts." % len(test_texts_1))

    tokenizer = Tokenizer(num_words=nb_words)
    tokenizer.fit_on_texts(texts_1 + test_texts_1)
    sequences_1 = tokenizer.texts_to_sequences(texts_1)
    word_index = tokenizer.word_index
    print("Found %s unique tokens." % len(word_index))

    test_sequences_1 = tokenizer.texts_to_sequences(test_texts_1)

    xtrain_pad = pad_sequences(sequences_1, maxlen=max_len)
    xtest_pad = pad_sequences(test_sequences_1, maxlen=max_len)
    del test_sequences_1
    del sequences_1
    nb_words_cnt = min(nb_words, len(word_index)) + 1

    # we need to binarize the labels for the neural net
    ytrain_enc = np_utils.to_categorical(Y_train)

    kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2017)
    pred_full_test = 0
    pred_train = np.zeros([xtrain_pad.shape[0], 3])
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=0, verbose=0, mode="auto"
    )
    for dev_index, val_index in kf.split(xtrain_pad):
        dev_X, val_X = xtrain_pad[dev_index], xtrain_pad[val_index]
        dev_y, val_y = ytrain_enc[dev_index], ytrain_enc[val_index]
        model = initNN(nb_words_cnt, max_len)
        model.fit(
            dev_X,
            y=dev_y,
            batch_size=32,
            epochs=4,
            verbose=1,
            validation_data=(val_X, val_y),
            callbacks=[early_stopping],
        )
        pred_val_y = model.predict(val_X)
        pred_test_y = model.predict(xtest_pad)
        pred_full_test = pred_full_test + pred_test_y
        pred_train[val_index, :] = pred_val_y
    return doAddNN(X_train, X_test, pred_train, pred_full_test / 5)
