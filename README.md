## TODO

topics

## Evaluation

log loss



Look at the characteristic words, 

check the topic used by the author, 

look how long are sentences, 

check the way of using commas.



## XGBoost0

**score: 0.28889, rank:45**

**score: 0.28895, rank: 45**

LDA Topic Model

**score: 0.29064, rank: 49**

XGBoost1 + XGBoost2

## XGBoost1

**score: 0.29229, rank: 55**

xgb:

* three nn
  * fastText
  * glove
  * nn
* word2vec: 100
* nb:
  * count words
  * count chars
  * tfidf words
  * tfidf chars

## XGBoost2

**score: 0.32343, rank: 121**

1. Meta features - features that are extracted from the text like number of words, number of stop words, number of punctuations etc
   1. Number of words in the text
   2. Number of unique words in the text
   3. Number of characters in the text
   4. Number of stopwords
   5. Number of punctuations
   6. Number of upper case words
   7. Number of title case words
   8. Average length of the words
2. Text based features - features directly based on the text / words like frequency, svd, word2vec etc.
   1. Naive Bayes on Word Tfidf Vectorizer
   2. Naive Bayes on Word Count Vectorizer
   3. SVD on word TFIDF
   4. Naive Bayes on Character Count Vectorizer
   5. Naive Bayes on Character Tfidf Vectorizer
   6. SVD on Character TFIDF

```
## Number of words in the text ##
train_df["num_words"] = train_df["text"].apply(lambda x: len(str(x).split()))
test_df["num_words"] = test_df["text"].apply(lambda x: len(str(x).split()))

train_df['num_words'].loc[train_df['num_words']>80] = 80 #truncation for better visuals
plt.figure(figsize=(12,8))
sns.violinplot(x='author', y='num_words', data=train_df)
plt.xlabel('Author Name', fontsize=12)
plt.ylabel('Number of words in text', fontsize=12)
plt.title("Number of words by author", fontsize=15)
plt.show()
```

```
### Plot the important variables ###
fig, ax = plt.subplots(figsize=(12,12))
xgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)
plt.show()
```

```
n_comp = 20
svd_obj = TruncatedSVD(n_components=n_comp, algorithm='arpack')
svd_obj.fit(full_tfidf)
train_svd = pd.DataFrame(svd_obj.transform(train_tfidf))
test_svd = pd.DataFrame(svd_obj.transform(test_tfidf))
    
train_svd.columns = ['svd_word_'+str(i) for i in range(n_comp)]
test_svd.columns = ['svd_word_'+str(i) for i in range(n_comp)]
train_df = pd.concat([train_df, train_svd], axis=1)
test_df = pd.concat([test_df, test_svd], axis=1)
del full_tfidf, train_tfidf, test_tfidf, train_svd, test_svd
```

```
cols_to_drop = ['id', 'text']
train_X = train_df.drop(cols_to_drop+['author'], axis=1)
test_X = test_df.drop(cols_to_drop, axis=1)

kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2017)
cv_scores = []
pred_full_test = 0
pred_train = np.zeros([train_df.shape[0], 3])
for dev_index, val_index in kf.split(train_X):
    dev_X, val_X = train_X.loc[dev_index], train_X.loc[val_index]
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    pred_val_y, pred_test_y, model = runXGB(dev_X, dev_y, val_X, val_y, test_X, seed_val=0, colsample=0.7)
    pred_full_test = pred_full_test + pred_test_y
    pred_train[val_index,:] = pred_val_y
    cv_scores.append(metrics.log_loss(val_y, pred_val_y))
    break
print("cv scores : ", cv_scores)

out_df = pd.DataFrame(pred_full_test)
out_df.columns = ['EAP', 'HPL', 'MWS']
out_df.insert(0, 'id', test_id)
out_df.to_csv("sub_fe.csv", index=False)
```

## Sklearn

## Bagging Classifier

```
from sklearn.ensemble import BaggingClassifier
m = BaggingClassifier(MultinomialNB(alpha=0.03))
m.fit(X_train, y_train)
m.predict_proba(X_test)
```

### Voting Classifier

**score: 0.33457, rank: 195**

```
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

models = [('MultiNB', MultinomialNB(alpha=0.03)),
          ('Calibrated MultiNB', CalibratedClassifierCV(
              MultinomialNB(alpha=0.03), method='isotonic')),
          ('Calibrated BernoulliNB', CalibratedClassifierCV(
              BernoulliNB(alpha=0.03), method='isotonic')),
          ('Calibrated Huber', CalibratedClassifierCV(
              SGDClassifier(loss='modified_huber', alpha=1e-4,
                            max_iter=10000, tol=1e-4), method='sigmoid')),
          ('Logit', LogisticRegression(C=30))]

train = pd.read_csv('./input/train.csv')
vectorizer=TfidfVectorizer(token_pattern=r'\w{1,}', sublinear_tf=True, ngram_range=(1,2))
clf = VotingClassifier(models, voting='soft', weights=[3,3,3,1,1])
X_train = vectorizer.fit_transform(train.text.values)
authors = ['MWS','EAP','HPL']
y_train = train.author.apply(authors.index).values
clf.fit(X_train, y_train)

test = pd.read_csv('./input/test.csv', index_col=0)
X_test = vectorizer.transform(test.text.values)
results = clf.predict_proba(X_test)
pd.DataFrame(results, index=test.index, columns=authors).to_csv('voting.csv')
```

## Keras

### Fast Text

**score: 0.36351, rank: 308**

* {lower: False, maxlen: 256}: 0.36351
* {lower: True, maxlen: 128}: 0.37240
* {lower: True, maxlen: 256}: 0.36897

```
tokenizer = Tokenizer(lower=False, filters='')
maxlen = 256
```

    def create_model(embedding_dims=20, optimizer='adam'):
        model = Sequential()
        model.add(Embedding(input_dim=input_dim, output_dim=embedding_dims))
        model.add(GlobalAveragePooling1D())
        model.add(Dense(3, activation='softmax'))
    
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])
        return model
```
Epoch 17/25
15663/15663 [==============================] - 8s 521us/step - loss: 0.0449 - acc: 0.9937 - val_loss: 0.3519 - val_acc: 0.8634
```

### NN1

```python
# get a list of classifications and generate numeric 
#  values for each class.  put the numeric class back 
#  on to the data frame.
authors = dict([(auth, idx) for idx, auth in enumerate(df['author'].unique())])
print(authors)
df['author_id'] = df['author'].apply(lambda x: authors[x])

df.head()
```

```
MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
```

```
# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

print('Training model.')

# train a 1D convnet with global maxpooling
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(128, 5, activation='relu')(embedded_sequences)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = GlobalMaxPooling1D()(x)
x = Dense(128, activation='relu')(x)
preds = Dense(len(authors), activation='softmax')(x)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])
```

```
Epoch 50/50
15664/15664 [==============================] - 2s 147us/step - loss: 0.0332 - acc: 0.9913 - val_loss: 2.9418 - val_acc: 0.6439
```

### NN2

```
MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 300
```

```
# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=True)

print('Training model.')

# train a 1D convnet with global maxpooling
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(128, 5, activation='relu')(embedded_sequences)
#x = MaxPooling1D()(x)
#x = Conv1D(128, 5, activation='relu')(x)
x = GlobalMaxPooling1D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(.5)(x)
preds = Dense(len(authors), activation='softmax')(x)
rms = RMSprop(lr=0.003)
model = Model(sequence_input, preds)
model.compile(loss='mean_squared_logarithmic_error',
              optimizer=rms, #'rmsprop',
              metrics=['acc'])
#model.compile(loss='categorical_crossentropy',
#              optimizer=rms, #'rmsprop',
#              metrics=['acc'])
```

```
Epoch 50/50
15664/15664 [==============================] - 5s 344us/step - loss: 0.0018 - acc: 0.9943 - val_loss: 0.0573 - val_acc: 0.8179
```

## NaiveBayesClassifier

```
# Add some of the shuffled terms as negative examples for each of the data samples.
allw = len(all_words)
idx = 0 # we are going to loop trough the shuffled values.
for passage in all_data:
    sample = list(passage[0].keys())
    j = 0
    #print(sample) 
    while j < len(sample): #  add the same number of negative samples as positive.
        current = all_words[shuffled_word_idxs[idx]]
        #print(current)
        if current not in sample:
            #  add the current term as a negative sample
            passage[0][current] = False
            ## increment j
            j = j+1
        ## increment index counter
        idx = idx+1
        if idx == allw:
            idx = 0 # reset and go around again
                
print(all_data[1])
```

```python
classifier = nltk.NaiveBayesClassifier.train(train_data)
classifier.show_most_informative_features()
preds = [classifier.classify(test) for test in test_data_stripped]
```

```
print(accuracy)  # 0.5742017879948914
```

