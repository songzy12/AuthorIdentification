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
