"""
Prediction on Validation sets
=============================

This example use the 'diabetes' database and
peridict the estimators on extracted validation sets from CV.

# Authors:  Kimia Nazarzadeh <k.nazarzadeh@fz-juelich.de>
            Kaustubh Patil <k.patil@fz-juelich.de>
#
# License: AGPL
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RepeatedKFold
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score

from julearn import run_cross_validation
from julearn.utils import configure_logging

###############################################################################
# Set the logging level to info to see extra information
configure_logging(level='INFO')

###############################################################################
# load the diabetes data from sklearn as a pandas dataframe
features, target = load_diabetes(return_X_y=True, as_frame=True)

###############################################################################
# Dataset contains ten variables age, sex, body mass index, average  blood
# pressure, and six blood serum measurements (s1-s6) diabetes patients and
# a quantitative measure of disease progression one year after baseline which
# is the target we are interested in predicting.

print('Features: \n', features.head())
print('Target: \n', target.describe())

###############################################################################
# Let's combine features and target together in one dataframe and define X
# and y
data_diabetes = pd.concat([features, target], axis=1)

X = ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
y = 'target'

###############################################################################
# Split the dataset into train and test
train_diabetes, test_diabetes = train_test_split(
    data_diabetes, test_size=0.1, random_state=200)

###############################################################################
# Model parameters for grid search
lambdas_ridge = [0.001, 0.01, 0.1,
                 1, 5, 10, 100, 1000, 10000, 100000]
c_vals_svm = np.geomspace(1e-2, 1e2, 5)
model_grid = {
    "linear_svm": [
        "svm",
        {
            "svm__kernel": ["linear"],
            "svm__C": c_vals_svm,
            'search': 'grid'
        },
    ],
}

model_name = "linear_svm"
model, model_params = model_grid[model_name]

###############################################################################
# Define number of splits (folds) and repeats for CV:
n_repeats = 5
n_folds = 5
cv = RepeatedKFold(n_splits=n_folds, n_repeats=n_repeats, random_state=200)
###############################################################################
# Train linear svm model with on training set:
# To evaluate the scores on the training set,
# We need to set 'return_train_score' to True.
# If difference between test score and training score is small,
# It means that the good model is good/good fit.
# Set return_estimator to 'all' to access all estimators.
scores, model_final = run_cross_validation(
    X=X, y=y, data=train_diabetes, preprocess_X='zscore', cv=cv,
    problem_type='regression', seed=200,
    model=model, model_params=model_params, scoring='r2',
    return_estimator='all', return_train_score=True)

###############################################################################
# The scores dataframe has all the values for each CV split.
print(scores.head())

###############################################################################
# Now we can get the estimator per fold and repetition:
df_estimators = scores.set_index(
    ['repeat', 'fold'])['estimator'].unstack()
df_estimators.index.name = 'Repeats'
df_estimators.columns.name = 'K-fold splits'

print(df_estimators)

###############################################################################
# Now we can get the test_score per fold and repetition:
df_scores = scores.set_index(
    ['repeat', 'fold'])['test_score'].unstack()
df_scores.index.name = 'Repeats'
df_scores.columns.name = 'K-fold splits'

print(df_scores)

###############################################################################
# Predict each estimator on validation set per fold:
# To access train and validation sets, using split function on CV object.
# The result must be the same as the 'test_score' of run_cross_validation.
df_prediction_scores = pd.DataFrame()

for idx, (train_val_index, validation_index) \
        in enumerate(cv.split(train_diabetes)):
    repeat = scores['repeat'][idx]
    fold = scores['fold'][idx]
    estimator = scores['estimator'][idx]
    y_pred = pd.Series(
        estimator.predict(train_diabetes.iloc[validation_index][X]))
    y_true = train_diabetes.iloc[validation_index][y]
    # use 'r2_score' scoring
    score = r2_score(y_true, y_pred)
    df_prediction_scores.loc[f'{repeat}', f'{fold}'] = score
df_prediction_scores.index.name = 'Repeats'
df_prediction_scores.columns.name = 'K-fold splits'

print(df_prediction_scores)

###############################################################################
# Visualization on how the prediction values and true values comparison.
fig, axes = plt.subplots(n_repeats, n_folds,
                         sharex=True, sharey=True, figsize=(20, 20))
for idx, (train_val_index, validation_index) \
        in enumerate(cv.split(train_diabetes)):
    repeat = scores['repeat'][idx]
    fold = scores['fold'][idx]
    estimator = scores['estimator'][idx]
    y_pred = pd.Series(
        estimator.predict(train_diabetes.iloc[validation_index][X]))
    y_true = train_diabetes.iloc[validation_index][y]
    score = r2_score(y_true, y_pred)
    sns.regplot(x=y_true, y=y_pred, ax=axes[repeat, fold], color='blue')
    xmin, xmax = axes[repeat, fold].get_xlim()
    ymin, ymax = axes[repeat, fold].get_ylim()
    axes[repeat, fold].set_xlim(xmin, xmax)
    axes[repeat, fold].set_ylim(ymin, ymax)
    # set labels
    axes[repeat, fold].set(xlabel=None, ylabel=None)
    axes[repeat, fold].tick_params(axis='both', labelsize=15)
    # axes[repeat, fold].text(0.5, 0.5, '$R^2$= {}'.format(score))
    axes[repeat, fold].text(.05, .96, '$r^2$={:.2f}'.format(score),
                            transform=axes[repeat, fold].transAxes,
                            va='top', ha='left', fontsize=12)
fig.text(0.5, 0.07, 'True Value', ha='center', fontsize=30)
fig.text(0.07, 0.5, 'Predicted Value', va='center',
         rotation='vertical', fontsize=30)
plt.suptitle('True vs Predicted Values',
             fontsize=30, y=.91)
