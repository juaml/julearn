"""
Custom Scoring Function for Regression
======================================

This example uses the ``diabetes`` data from ``sklearn datasets`` and performs
a regression analysis using a Ridge Regression model. As scorers, it uses
``scikit-learn``, ``julearn`` and a custom metric defined by the user.

"""
# Authors: Shammi More <s.more@fz-juelich.de>
#          Federico Raimondo <f.raimondo@fz-juelich.de>
# License: AGPL

import pandas as pd
import scipy
from sklearn.datasets import load_diabetes

from sklearn.metrics import make_scorer
from julearn.scoring import register_scorer

from julearn import run_cross_validation
from julearn.utils import configure_logging

###############################################################################
# Set the logging level to info to see extra information.
configure_logging(level="INFO")

###############################################################################
# load the diabetes data from ``sklearn`` as a ``pandas.DataFrame``.
features, target = load_diabetes(return_X_y=True, as_frame=True)

###############################################################################
# Dataset contains ten variables age, sex, body mass index, average blood
# pressure, and six blood serum measurements (s1-s6) diabetes patients and
# a quantitative measure of disease progression one year after baseline which
# is the target we are interested in predicting.
print("Features: \n", features.head())
print("Target: \n", target.describe())

###############################################################################
# Let's combine features and target together in one dataframe and define X
# and y.
data_diabetes = pd.concat([features, target], axis=1)  # type: ignore

X = ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]
y = "target"

###############################################################################
# Train a ridge regression model on train dataset and use mean absolute error
# for scoring.
scores, model = run_cross_validation(
    X=X,
    y=y,
    data=data_diabetes,
    preprocess="zscore",
    problem_type="regression",
    model="ridge",
    return_estimator="final",
    scoring="neg_mean_absolute_error",
)

###############################################################################
# The scores dataframe has all the values for each CV split.
scores.head()

###############################################################################
# Mean value of mean absolute error across CV.
print(scores["test_score"].mean() * -1)

###############################################################################
# Now do the same thing, but use mean absolute error and Pearson product-moment
# correlation coefficient (squared) as scoring functions.
scores, model = run_cross_validation(
    X=X,
    y=y,
    data=data_diabetes,
    preprocess="zscore",
    problem_type="regression",
    model="ridge",
    return_estimator="final",
    scoring=["neg_mean_absolute_error", "r2_corr"],
)

###############################################################################
# Now the scores dataframe has all the values for each CV split, but two scores
# unders the column names ``"test_neg_mean_absolute_error"`` and
# ``"test_r2_corr"``.
print(scores[["test_neg_mean_absolute_error", "test_r2_corr"]].mean())

###############################################################################
# If we want to define a custom scoring metric, we need to define a function
# that takes the predicted and the actual values as input and returns a value.
# In this case, we want to compute Pearson correlation coefficient (r).


def pearson_scorer(y_true, y_pred):
    return scipy.stats.pearsonr(y_true.squeeze(), y_pred.squeeze())[0]


###############################################################################
# Before using it, we need to convert it to a ``sklearn scorer`` and register it
# with ``julearn``.

register_scorer(scorer_name="pearsonr", scorer=make_scorer(pearson_scorer))

###############################################################################
# Now we can use it as another scoring metric.
scores, model = run_cross_validation(
    X=X,
    y=y,
    data=data_diabetes,
    preprocess="zscore",
    problem_type="regression",
    model="ridge",
    return_estimator="final",
    scoring=["neg_mean_absolute_error", "r2_corr", "pearsonr"],
)
