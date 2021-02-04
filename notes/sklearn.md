## Bagging Classifier

```
from sklearn.ensemble import BaggingClassifier
m = BaggingClassifier(MultinomialNB(alpha=0.03))
m.fit(X_train, y_train)
m.predict_proba(X_test)
```

## Voting Classifier

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
