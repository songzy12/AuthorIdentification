import numpy as np
from sklearn import metrics, model_selection

import xgboost as xgb


def run_xgb(
    train_X,
    train_y,
    test_X,
    test_y=None,
    test_X2=None,
    seed_val=0,
    child=1,
    colsample=0.3,
):
    param = {}
    param["objective"] = "multi:softprob"
    param["eta"] = 0.1
    param["max_depth"] = 3
    param["silent"] = 1
    param["num_class"] = 3
    param["eval_metric"] = "mlogloss"
    param["min_child_weight"] = child
    param["subsample"] = 0.8
    param["colsample_bytree"] = colsample
    param["seed"] = seed_val
    num_rounds = 2000

    plst = list(param.items())
    xgtrain = xgb.DMatrix(train_X, label=train_y)

    if test_y is not None:
        xgtest = xgb.DMatrix(test_X, label=test_y)
        watchlist = [(xgtrain, "train"), (xgtest, "test")]
        model = xgb.train(
            plst,
            xgtrain,
            num_rounds,
            watchlist,
            early_stopping_rounds=50,
            verbose_eval=20,
        )
    else:
        xgtest = xgb.DMatrix(test_X)
        model = xgb.train(plst, xgtrain, num_rounds)

    pred_test_y = model.predict(xgtest, ntree_limit=model.best_ntree_limit)
    if test_X2 is not None:
        xgtest2 = xgb.DMatrix(test_X2)
        pred_test_y2 = model.predict(xgtest2, ntree_limit=model.best_ntree_limit)
    return pred_test_y, pred_test_y2, model


def run_kfold_xgb(train_X, train_y, test_X, n_splits=5):
    kf = model_selection.KFold(n_splits=n_splits, shuffle=True, random_state=2017)
    cv_scores = []
    pred_full_test = 0
    pred_train = np.zeros([train_X.shape[0], 3])
    for dev_index, val_index in kf.split(train_X):
        dev_X, val_X = train_X.loc[dev_index], train_X.loc[val_index]
        dev_y, val_y = train_y[dev_index], train_y[val_index]
        pred_val_y, pred_test_y, model = run_xgb(
            dev_X, dev_y, val_X, val_y, test_X, seed_val=0, colsample=0.7
        )
        pred_full_test = pred_full_test + pred_test_y
        pred_train[val_index, :] = pred_val_y
        cv_scores.append(metrics.log_loss(val_y, pred_val_y))
    print("cv scores : ", cv_scores)

    pred_full_test /= 5.0
    return pred_full_test
