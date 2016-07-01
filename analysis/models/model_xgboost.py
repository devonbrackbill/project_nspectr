from __future__ import division
import numpy as np
import pandas as pd
#from __future__ import division
#%matplotlib inline 
import matplotlib.pyplot as plt
from sklearn import linear_model
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import auc

train = pd.read_csv('../../finalData/train_sub200_all.csv')
test = pd.read_csv('../../finalData/test_sub200_all.csv')

train_y = train['vio3']
test_y = test['vio3']

train_X = train[train.columns[1:201]]
test_X = test[test.columns[1:201]]

gbm = xgb.XGBClassifier()
gbm_params = {
    'learning_rate': [0.05, 0.1],
    'n_estimators': [300, 500],
    'max_depth': [2, 3, 8],
}
cv = StratifiedKFold(train_y)
grid = GridSearchCV(gbm, gbm_params,scoring='roc_auc',cv=cv,verbose=10,n_jobs=7)
grid.fit(train_X, train_y)

print (grid.best_params_)

preds_probs = grid.best_estimator_.predict_proba(test_X)
preds_raw = grid.best_estimator_.predict(test_X)
print(sum(preds_raw == test_y)/len(test_y))

from sklearn import metrics
print(metrics.roc_auc_score(y_true = test_y, y_score = preds_probs[:,1]))
