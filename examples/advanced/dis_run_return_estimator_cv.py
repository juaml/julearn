"""
Regression Analysis with returning CV Estimator.
================================================

This example uses the 'diabetes' data from sklearn datasets
and performs a regression analysis with returning CV estimator
using a Ridge Regression model.

"""

# Authors: Kimia Nazarzadeh <k.nazarzadeh@fz-juelich.de>
#          Lya K. Paas Oliveros <l.paas.oliveros@fz-juelich.de>
#
# License: AGPL

import pandas as pd

from sklearn.datasets import load_diabetes
from sklearn.model_selection import RepeatedKFold

from julearn import run_cross_validation
from julearn.pipeline import PipelineCreator
from julearn.utils import configure_logging
from julearn.inspect import preprocess

###############################################################################
# Set the logging level to info to see extra information
configure_logging(level="INFO")

###############################################################################
# load the diabetes data from sklearn as a pandas dataframe
features, target = load_diabetes(return_X_y=True, as_frame=True)

###############################################################################
# Dataset contains ten variables age, sex, body mass index, average  blood
# pressure, and six blood serum measurements (s1-s6) diabetes patients and
# a quantitative measure of disease progression one year after baseline which
# is the target we are interested in predicting.

print("Features: \n", features.head())
print("Target: \n", target.describe())

###############################################################################
# Let's combine features and target together in one dataframe and define X
# and y
data_diabetes = pd.concat([features, target], axis=1)

X = ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]
y = "target"

###############################################################################
# Repeated K-Fold cross validator n times
# with different randomization in each repetition
cv = RepeatedKFold(n_splits=3, n_repeats=3, random_state=1)

###############################################################################
# Create pipeline for regression problem type
creator = PipelineCreator(problem_type="regression")
creator.add("zscore", apply_to="*")
creator.add("ridge")

###############################################################################
# Train a ridge regression model on diabetes dataset
# and use mean absolute error for scoring.
# With return estimator CV it returns all models on each fold
scores = run_cross_validation(
    X=X,
    y=y,
    data=data_diabetes,
    model=creator,
    seed=1,
    cv=cv,
    return_estimator="cv",
    scoring="neg_mean_absolute_error"
)

###############################################################################
# The scores dataframe has all the values for each CV split.
print(scores.head())

###############################################################################
# Explore the model that was used for a single fold
# for example, printing the first estimator which is
# stored in the first line of scores.
estimator_1 = scores['estimator'][0]
print(estimator_1)

###############################################################################
# Apply the first model of CV to the entire dataset
preprocess(estimator_1, X, data_diabetes)

###############################################################################
# Recreate the first fold. First, we will store the cv generator
splitter = cv.split(data_diabetes)

# With next, we obtain the indices for train
# and test set in the first fold.
# If you would run the same line again,
# you would obtain the indices for the second fold.
train_idx, test_idx = next(splitter)

# Let's see the train and test set for the first fold
print(data_diabetes.iloc[train_idx, :])
print(data_diabetes.iloc[test_idx, :])

# And finally, we apply the first estimator to the train
# and test set of the first fold
print(preprocess(estimator_1, X, data_diabetes.iloc[train_idx, :]))
print(preprocess(estimator_1, X, data_diabetes.iloc[test_idx, :]))
