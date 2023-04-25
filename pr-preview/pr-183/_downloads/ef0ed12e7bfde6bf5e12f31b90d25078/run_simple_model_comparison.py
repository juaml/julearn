"""
Simple Model Comparison
=======================

This example uses the 'iris' dataset and performs binary classifications
using different models. At the end, it compares the performance of the models
using different scoring functions and performs a statistical test to assess
whether the difference in performance is significant.

.. include:: ../../links.inc
"""
# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#
# License: AGPL

from seaborn import load_dataset
from sklearn.model_selection import RepeatedStratifiedKFold
from julearn import run_cross_validation
from julearn.utils import configure_logging
from julearn.stats.corrected_ttest import corrected_ttest

###############################################################################
# Set the logging level to info to see extra information
configure_logging(level="INFO")

###############################################################################
df_iris = load_dataset("iris")

###############################################################################
# The dataset has three kind of species. We will keep two to perform a binary
# classification.
df_iris = df_iris[df_iris["species"].isin(["versicolor", "virginica"])]

###############################################################################
# As features, we will use the sepal length, width and petal length.
# We will try to predict the species.

X = ["sepal_length", "sepal_width", "petal_length"]
y = "species"
scores = run_cross_validation(
    X=X,
    y=y,
    data=df_iris,
    model="svm",
    problem_type="classification",
    preprocess="zscore",
)

print(scores["test_score"])

###############################################################################
# Additionally, we can choose to assess the performance of the model using
# different scoring functions.
#
# For example, we might have an unbalanced dataset:

df_unbalanced = df_iris[20:]  # drop the first 20 versicolor samples
print(df_unbalanced["species"].value_counts())

###############################################################################
# So we will choose to use the `balanced_accuracy` and `roc_auc` metrics.
#
scoring = ["balanced_accuracy", "roc_auc"]

###############################################################################
# Since we are comparing the performance of different models, we will need
# to use the same random seed to split the data in the same way.

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)

###############################################################################
# First we will use a default SVM model.
scores1 = run_cross_validation(
    X=X,
    y=y,
    data=df_unbalanced,
    model="svm",
    preprocess="zscore",
    problem_type="classification",
    scoring=scoring,
    cv=cv,
)

scores1["model"] = "svm"

###############################################################################
# Second we will use a default Random Forest model.
scores2 = run_cross_validation(
    X=X,
    y=y,
    data=df_unbalanced,
    model="rf",
    preprocess="zscore",
    problem_type="classification",
    scoring=scoring,
    cv=cv,
)

scores2["model"] = "rf"

###############################################################################
# The third model will be a SVM with a linear kernel.
scores3 = run_cross_validation(
    X=X,
    y=y,
    data=df_unbalanced,
    model="svm",
    model_params={"svm__kernel": "linear"},
    preprocess="zscore",
    problem_type="classification",
    scoring=scoring,
    cv=cv,
)

scores3["model"] = "svm_linear"

stats_df = corrected_ttest(scores1, scores2, scores3)
print(stats_df)