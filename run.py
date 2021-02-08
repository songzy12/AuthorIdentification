import pandas as pd

from feature_extractor.text_feature_extractor import add_text_feature
from feature_extractor.embedding_feature_extractor import add_glove_feature
from feature_extractor.topic_feature_extractor import add_lda_feature
from feature_extractor.topic_feature_extractor import add_svd_feature
from model.cnn import doNN
from model.fast_text import doFastText
from model.glove_dnn import doNN_glove
from model.naive_bayes import run_mnbs
from model.xgboost import run_kfold_xgb

# Read the train and test dataset and check the top few lines.
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

print("train_df.shape: ", train_df.shape)
print("test_df.shape: ", test_df.shape)

train_df = add_text_feature(train_df)
test_df = add_text_feature(test_df)
print("Text features added...")

train_df, test_df = add_lda_feature(train_df, test_df)
train_df, test_df = add_svd_feature(train_df, test_df)
print("LDA and SVD features added...")

train_df, test_df, train_df_glove, test_df_glove, word_vecs = \
    add_glove_feature(train_df, test_df)
print("GloVe features added...")

author_mapping_dict = {"EAP": 0, "HPL": 1, "MWS": 2}
train_y = train_df["author"].map(author_mapping_dict)

train_df, test_df = run_mnbs(train_df, test_df, train_y)
print("MNB finished...")

train_df, test_df = doNN_glove(
    train_df, test_df, train_y, train_df_glove, test_df_glove
)
print("GloVe DNN finished...")

train_df, test_df = doFastText(train_df, test_df, train_y)
print("FastText finished...")

train_df, test_df = doNN(train_df, test_df, train_y)
print("CNN finished...")

train_df.to_pickle("./output/train_df.pkl")
test_df.to_pickle("./output/test_df.pkl")

cols_to_drop = ["id", "text"]
train_df = train_df.drop(columns=cols_to_drop + ["author"])
test_df = test_df.drop(columns=cols_to_drop)

pred_full_test = run_kfold_xgb(train_df, test_df, train_y)
print("XGBoost finished...")

out_df = pd.DataFrame(pred_full_test)
out_df.columns = ["EAP", "HPL", "MWS"]
out_df.insert(0, "id", test_df["id"].values)
out_df.to_csv("./output/submission.csv", index=False)
