# coding: utf-8
from __future__ import division
import numpy as np
import pandas as pd
get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics

train = pd.read_csv('../../finalData/train_sub200_all.csv')
test = pd.read_csv('../../finalData/test_sub200_all.csv')

train_y = train['vio3']
test_y = test['vio3']

# start from column 1 instead of 0 b/c 0 is the index
train_X = train[train.columns[1:201]]
test_X = test[test.columns[1:201]]

# CLASSIFIER
# need to set max_iter high because default settings do not converge;
#also change the solver b/c there is an error with machine's liblinear package (?)
logistic = linear_model.LogisticRegression(solver='sag', max_iter=100000)
logistic.fit(train_X,train_y)

preds_logistic_probs = logistic.predict_proba(test_X)
preds_logistic_raw= logistic.predict(test_X)
sum(preds_logistic_raw == test_y)/len(test_y)

print(metrics.roc_auc_score(y_true = test_y, y_score = preds_logistic_probs[:,1]))

# use grid search to select the severity of the l2 penalty
logistic2 = linear_model.LogisticRegressionCV(cv=10,penalty='l2',
                                              solver = 'sag',
                                              max_iter=100000,
                                              n_jobs = 7,
                                              verbose = 1,
                                              refit = True)

logistic2.fit(train_X,train_y)

preds_logistic2_probs = logistic2.predict_proba(test_X)
preds_logistic2_raw= logistic2.predict(test_X)

print(sum(preds_logistic2_raw == test_y)/len(test_y))
print(metrics.roc_auc_score(y_true = test_y, y_score = preds_logistic2_probs[:,1]))