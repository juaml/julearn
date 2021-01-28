"""
Simple Binary Classification
============================

This example uses the 'iris' dataset and performs a simple binary
classification using a Support Vector Machine classifier.

.. include:: ../../links.inc
"""
# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#
# License: AGPL
from seaborn import load_dataset
from julearn import run_cross_validation
from julearn.utils import configure_logging

###############################################################################
# Set the logging level to info to see extra information
configure_logging(level='INFO')

###############################################################################
df_iris = load_dataset('iris')

###############################################################################
# The dataset has three kind of species. We will keep two to perform a binary
# classification.
df_iris = df_iris[df_iris['species'].isin(['versicolor', 'virginica'])]

###############################################################################
# As features, we will use the sepal length, width and petal length.
# We will try to predict the species.

X = ['sepal_length', 'sepal_width', 'petal_length']
y = 'species'
scores = run_cross_validation(
    X=X, y=y, data=df_iris, model='svm', preprocess_X='zscore')

print(scores['test_score'])

###############################################################################
# Additionally, we can choose to assess the performance of the model using
# different scoring functions.
#
# For example, we might have an unbalanced dataset:

df_unbalanced = df_iris[20:]  # drop the first 20 versicolor samples
print(df_unbalanced['species'].value_counts())

###############################################################################
# If we compute the `accuracy`, we might not account for this imbalance. A more
# suitable metric is the `balanced_accuracy`. More information in scikit-learn:
# `Balanced Accuracy`_
#
# We will also set the random seed so we always split the data in the same way.
scores = run_cross_validation(
    X=X, y=y, data=df_unbalanced, model='svm', seed=42, preprocess_X='zscore',
    scoring=['accuracy', 'balanced_accuracy'])

print(scores['test_accuracy'].mean())
print(scores['test_balanced_accuracy'].mean())


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
precision_scores = run_cross_validation(
    X=X, y=y, data=df_unbalanced, model='svm', preprocess_X='zscore', seed=42,
    scoring='precision', pos_labels='versicolor')
print(precision_scores['test_score'].mean())
