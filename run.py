import pandas as pd


from feature_extractor.text_feature_extractor import add_text_feature
from feature_extractor.embedding_feature_extractor import add_glove_feature
from model.neural_network import doNN, doFastText, doNN_glove
from model.xgboost import run_kfold_xgb

# Read the train and test dataset and check the top few lines #
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
print("Number of rows in train dataset : ", train_df.shape[0])
print("Number of rows in test dataset : ", test_df.shape[0])

# Prepare the data for modeling ##
author_mapping_dict = {"EAP": 0, "HPL": 1, "MWS": 2}
train_y = train_df["author"].map(author_mapping_dict)

for df in [train_df, test_df]:
    df = add_text_feature(df)


glove_vecs_train, glove_vecs_test, embeddings_index = add_glove_feature(
    train_df["text"], test_df["text"]
)
train_df[["sent_vec_" + str(i) for i in range(100)]] = pd.DataFrame(
    glove_vecs_train.tolist()
)
test_df[["sent_vec_" + str(i) for i in range(100)]] = pd.DataFrame(
    glove_vecs_test.tolist()
)
print("Glove sentence vector finished...")

train_df, test_df = doFastText(train_df, test_df, train_y)
print("FastText finished...")
train_df, test_df = doNN(train_df, test_df, train_y)
print("NN finished...")
train_df, test_df = doNN_glove(
    train_df, test_df, train_y, glove_vecs_train, glove_vecs_test
)
print("NN Glove finished...")

cols_to_drop = ["id", "text", "split", "sid", "pos_tag"]
train_X = train_df.drop(cols_to_drop + ["author"], axis=1)
test_X = test_df.drop(cols_to_drop, axis=1)

train_df.to_pickle("./output/train_df.pkl")
test_df.to_pickle("./output/test_df.pkl")

pred_full_test = run_kfold_xgb(train_X, train_y, test_X)

out_df = pd.DataFrame(pred_full_test)
out_df.columns = ["EAP", "HPL", "MWS"]
out_df.insert(0, "id", test_df["id"].values)
out_df.to_csv("./output/submission.csv", index=False)
