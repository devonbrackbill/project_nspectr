# project_nspectr
code for nspectr.org

# About

This is the model behind [nspectr.org](nspectr.org), an app that predicts restaurant violations in Boston.

The main steps are:

*clean the data using PrepData.R

*run models. This can be done either with the R models, or with the Python models, as they replicate the same analysis.

The R models use the H2O library, which is a distributed Java virtual machine that allows for efficient parallel computation of machine learning algorithms.

There are are 5 model files:

*model_feature_selection (.R only): runs cross-validation to reduce the number of features (initially 5,000+) down to the optimal number of 200.

*model_baseline (.R and .py): a random forest model

*model_logistic (.R and .py): a logistic regression with L2 regularization and a cross-validated grid search of $C$, the regularization parameter.

*model_xgboost  (.py only): a gradient boosted machine model (using trees) that examines a large grid of hyperparameters to optimize the GBM. In particular, I consider the learning rate (eta), the tree depth, and the number of trees to grow.

*model_xgboost2 (.py only): an additional search of the hyperparameter space after the results from the first grid search.

# Model performance:

The logistic regression performs worst. The random forest and gbm models are competitive, both achieving near 70% accuracy, and 0.8 AUC, and they weren't significantly different from each other on the validation set.
