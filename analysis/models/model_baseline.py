# coding: utf-8

# BASELINE RANDOM FOREST MODEL
from __future__ import division
import numpy as np
import pandas as pd
#get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn import metrics

# LOAD DATA
all_data = pd.DataFrame.from_csv('../../finalData/DataPriorToModels.csv')
all_data = all_data.drop(['restaurant_id', 'prior_date', 'open'],1,errors='ignore')

# simply drop all columns with na's
all_data = all_data.dropna(axis=1)

# TRAIN/TEST SPLIT 
train_X = all_data[all_data['year'] < 2014]
test_X = all_data[all_data['year'] >= 2014]

train_y = train_X['vio3']
test_y = test_X['vio3']

train_X = train_X.drop(['vio1', 'vio2', 'vio3'],axis=1, errors='ignore')
test_X = test_X.drop(['vio1', 'vio2', 'vio3'],axis=1, errors='ignore')

train_X = train_X.drop(['date'],1)
test_X = test_X.drop(['date'],1)

# the first column is a non-sense column from pre-processing
train_X = train_X.drop(train_X.columns[0],axis=1)
test_X = test_X.drop(test_X.columns[0], axis=1)

# create dummy var for severe violations (y3)
train_y_dum = train_y.map(lambda x: x>=1 and 1 or 0)
test_y_dum = test_y.map(lambda x: x>=1 and 1 or 0)

my_mod = RandomForestClassifier(
            n_estimators = 500,
            n_jobs = 6)

my_mod.fit(train_X,train_y_dum)

preds = my_mod.predict(test_X)

print(sum(preds == test_y_dum)/len(test_y_dum))

# VARIABLE IMPORTANCE
imp = my_mod.feature_importances_
importances = my_mod.feature_importances_
std = np.std([tree.feature_importances_ for tree in my_mod.estimators_],axis=0)
indices = np.argsort(importances)[::-1]

# SUBSET THE DATA TO THE TOP 200 FEATURES (200 WAS CHOSEN BY CROSS-VALIDATION IN: featureSelection.R)
indices = indices[0:200]

# Print the feature ranking
print("Feature ranking:")

for f in range(30):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

for i in train_X.columns[indices]:
    print i

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(len(indices)), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(20), train_X.columns[indices], rotation=90)
plt.xlim([-1, 20])
plt.show()


# SUBSET THE DATA TO THE TOP 200 FEATURES (200 WAS CHOSEN BY CROSS-VALIDATION IN: featureSelection.R)
train_X_sub = train_X[indices]
train_X_sub.shape

test_X_sub = test_X[indices]
test_X_sub.shape

# save 200-feature datasets to file for use in other models
train_X_sub.to_csv('../../finalData/train_sub200.csv')
test_X_sub.to_csv('../../finalData/test_sub200.csv')

train_y_dum.to_csv('../../finalData/train_y_dum.csv')
test_y_dum.to_csv('../../finalData/test_y_dum.csv')

# savle all 200-features + y value to file
train_X_sub_all = pd.concat([train_X_sub, train_y_dum], axis=1)
test_X_sub_all = pd.concat([test_X_sub, test_y_dum], axis = 1)

train_X_sub_all.to_csv('../../finalData/train_sub200_all.csv')
test_X_sub_all.to_csv('../../finalData/test_sub200_all.csv')

# RANDOM FOREST ON 200 FEATURES (this ends up being the final reported model for nspectr.org)
rf_mod = RandomForestClassifier(
            n_estimators = 500,
            n_jobs = 6)
rf_mod.fit(train_X_sub,train_y_dum)

preds_rf_probs = rf_mod.predict_proba(test_X_sub)
preds_rf_raw= rf_mod.predict(test_X_sub)

# these metrics are identical to those found using model_baseline.R
sum(preds_rf_raw == test_y_dum)/len(test_y_dum)

# same metrics as model_baseline.R
metrics.roc_auc_score(y_true = test_y_dum, y_score = preds_rf_probs[:,1])