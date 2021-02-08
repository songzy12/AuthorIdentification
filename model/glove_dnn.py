import numpy as np
from sklearn import model_selection
from sklearn import preprocessing

from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.normalization import BatchNormalization


# NN Glove
def doAddNN_glove(X_train, X_test, pred_train, pred_test):
    X_train["nn_glove_eap"] = pred_train[:, 0]
    X_train["nn_glove_hpl"] = pred_train[:, 1]
    X_train["nn_glove_mws"] = pred_train[:, 2]
    X_test["nn_glove_eap"] = pred_test[:, 0]
    X_test["nn_glove_hpl"] = pred_test[:, 1]
    X_test["nn_glove_mws"] = pred_test[:, 2]
    return X_train, X_test


def initNN_glove():
    # create a simple 3 layer sequential neural net
    model = Sequential()

    model.add(Dense(128, input_dim=100, activation="relu"))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())

    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())

    model.add(Dense(3))
    model.add(Activation("softmax"))

    # compile the model
    model.compile(loss="categorical_crossentropy", optimizer="adam")
    return model


def doNN_glove(X_train, X_test, Y_train, xtrain_glove, xtest_glove):
    # scale the data before any neural net:
    scl = preprocessing.StandardScaler()
    ytrain_enc = np_utils.to_categorical(Y_train)
    kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2017)

    pred_full_test = 0
    xtrain_glove = scl.fit_transform(xtrain_glove)
    xtest_glove = scl.fit_transform(xtest_glove)
    pred_train = np.zeros([xtrain_glove.shape[0], 3])
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=0, verbose=0, mode="auto"
    )

    for dev_index, val_index in kf.split(xtrain_glove):
        dev_X, val_X = xtrain_glove[dev_index], xtrain_glove[val_index]
        dev_y, val_y = ytrain_enc[dev_index], ytrain_enc[val_index]
        model = initNN_glove()
        model.fit(
            dev_X,
            y=dev_y,
            batch_size=32,
            epochs=10,
            verbose=1,
            validation_data=(val_X, val_y),
            callbacks=[early_stopping],
        )
        pred_val_y = model.predict(val_X)
        pred_test_y = model.predict(xtest_glove)
        pred_full_test = pred_full_test + pred_test_y
        pred_train[val_index, :] = pred_val_y
    return doAddNN_glove(X_train, X_test, pred_train, pred_full_test / 5)
