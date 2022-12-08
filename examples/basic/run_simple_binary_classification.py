""".

Simple Binary Classification
============================

This example uses the 'iris' dataset and performs a simple binary
classification using a Support Vector Machine classifier.

We demonstrate how to use the main julearn function "run_cross_validation".
We also show how to use some of the most basic features provided
by this function.

References
----------
Fisher, R. A. (1936). The use of multiple measurements in taxonomic problems.
Annals of eugenics, 7(2), 179-188.

.. include:: ../../links.inc
"""
# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Nicolás Nieto <n.nieto@fz-juelich.de>
# License: AGPL
from seaborn import load_dataset
from julearn import run_cross_validation
from julearn.utils import configure_logging
from sklearn.model_selection import StratifiedKFold
###############################################################################
# Set the logging level to info to see extra information
configure_logging(level='INFO')

###############################################################################
# Load iris dataset from `seaborn` library. It contains information about
# three flowers: `species`, `versicolor`,`virginica` and `setosa`.
# For each observation, it has four features: `sepal_length`, `sepal_width`,
# `petal_length` and `petal_width`.
#
# The data needs to be a `pandas.DataFrame`
df_iris = load_dataset('iris')

###############################################################################
# The dataset has three kind of species. We will keep only two of them
# to perform a binary classification.
df_iris = df_iris[df_iris['species'].isin(['versicolor', 'virginica'])]

###############################################################################
# As features, we will only use sepal length, width and petal length.
# We set a list with the name of the columns we want to use as features.
# Those are columns names of `df_iris`
X = ['sepal_length', 'sepal_width', 'petal_length']

###############################################################################

# As target, we will use the species, as we want to predict the specie with the
# information about the petal and sepal.
# We set a variable with the name of the column we want to use as a target.
# The target is also included in `df_iris`
y = 'species'

###############################################################################
# We provide the model name as a string. We will use a Support Vector Machine.
# For all available models see `models.available_models`_
# Additionally, as a standard step in a classification pipeline, we will
# z-score the data. We will provide the preprocess as a string.
model = 'svm'
preprocess = 'zscore'

###############################################################################
# If we take a look at the data, we will find that the samples are sorted by
# the target. In that case, a problem will occur if we use a simple Kfold
# approach, as some splits will contain only samples from one class.
#
# To solve this, we will use a Stratified KFold approach, that takes care of
# this, and makes sure that every split have the same proportion of data for
# each class.
cv = StratifiedKFold(n_splits=5)

###############################################################################
# 'run_cross_validation'is the main function in julearn.
# It will run a complete pipeline and return the test scores for all the folds.
# The function returns different information about the execution.
# We will print the mean test accuracy obtained in all folders (5 by default).
#
# Note that 'df_iris' contains all the information and X and y are only list of
# strings or just a strings with the column names used.
scores = run_cross_validation(
    X=X, y=y, data=df_iris, model=model, preprocess=preprocess, cv=cv)

print('Mean Test Accuracy: %f ' % scores['test_score'].mean())

###############################################################################
# Additionally, we can choose to assess the performance of the model using
# different scoring functions.
#
# For example, we might have an unbalanced dataset:

df_unbalanced = df_iris[20:]  # drop the first 20 ve10rsicolor samples
print('New number of samples for each class:')
print(df_unbalanced['species'].value_counts())


###############################################################################
# If we compute the `accuracy`, we might not account for this imbalance. A more
# suitable metric is the `balanced_accuracy`. More information in scikit-learn:
# `Balanced Accuracy`.
#
# We will also set the random seed so we always split the data in the same way.
# julearn is able to return several metrics. We need to pass the desired
# metrics in a list.
scores = run_cross_validation(
    X=X, y=y, data=df_unbalanced, model=model, seed=42, cv=cv,
    preprocess=preprocess, scoring=['accuracy', 'balanced_accuracy'])

print('Mean Accuracy: %f' % scores['test_accuracy'].mean())
print('Mean Balanced Accuracy: %f' % scores['test_balanced_accuracy'].mean())

###############################################################################
# Other kind of metrics allows us to evaluate how good our model is to detect
# specific targets. Suppose we want to create a model that correctly identifies
# the `versicolor` samples.
#
# Now we might want to evaluate the precision score, or the ratio of true
# positives (tp) over all positives (true and false positives). More
# information in scikit-learn: `Precision`_
#
# For this metric to work, we need to define which are our `positive` values.
# In this example, we are interested in detecting `versicolor`.
#
# Note that if only one scoring is required, the returned name is 'test_score'
# independently of which scoring was requested.

scores = run_cross_validation(
    X=X, y=y, data=df_unbalanced, model=model, preprocess=preprocess, cv=cv,
    seed=42, scoring='precision', pos_labels='versicolor')

print('Mean Precision %f ' % scores['test_score'].mean())
