## Fast Text

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

## NN1

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

## NN2

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
