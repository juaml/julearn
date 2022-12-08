"""
Stratified K-fold CV for regression analysis
============================================

This example uses the 'diabetes' data from sklearn datasets to
perform stratified Kfold CV for a regression problem,

.. include:: ../../links.inc
"""
# Authors: Shammi More <s.more@fz-juelich.de>
#          Federico Raimondo <f.raimondo@fz-juelich.de>
#
# License: AGPL

import math
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import KFold

from julearn import run_cross_validation
from julearn.utils import configure_logging
from julearn.model_selection import StratifiedGroupsKFold

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
# Let's combine features and target together in one dataframe and create some
# outliers to see the difference in model performance with and without
# stratification

data_df = pd.concat([features, target], axis=1)

# Create outliers for test purpose
new_df = data_df[(data_df.target > 145) & (data_df.target <= 150)]
new_df['target'] = [590, 580, 597, 595, 590, 590, 600]
data_df = pd.concat([data_df, new_df], axis=0)
data_df =  data_df.reset_index(drop=True)

# define X, y
X = ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
y = 'target'

###############################################################################
# Define number of splits for CV and create bins/group for stratification
num_splits = 7

num_bins = math.floor(len(data_df) / num_splits)  # num of bins to be created
bins_on = data_df.target  # variable to be used for stratification
qc = pd.cut(bins_on.tolist(), num_bins)  # divides data in bins
data_df['bins'] = qc.codes
groups = 'bins'

###############################################################################
# Train a linear regression model with stratification on target

cv_stratified = StratifiedGroupsKFold(n_splits=num_splits, shuffle=False)
scores_strat, model = run_cross_validation(
    X=X, y=y, data=data_df, preprocess_X='zscore', cv=cv_stratified,
    groups=groups, problem_type='regression', model='linreg',
    return_estimator='final', scoring='neg_mean_absolute_error')

###############################################################################
# Train a linear regression model without stratification on target

cv = KFold(n_splits=num_splits, shuffle=False, random_state=None)
scores, model = run_cross_validation(
    X=X, y=y, data=data_df, preprocess_X='zscore', cv=cv,
    problem_type='regression', model='linreg', return_estimator='final',
    scoring='neg_mean_absolute_error')

###############################################################################
# Now we can compare the test score for model trained with and without
# stratification. We can combine the two outputs as pandas dataframes

scores_strat['model'] = 'With stratification'
scores['model'] = 'Without stratification'
df_scores = scores_strat[['test_score', 'model']]
df_scores  = pd.concat([df_scores, scores[['test_score', 'model']]])

###############################################################################
# Plot a boxplot with test scores from both the models. We see here that
# the variance for the test score is much higher when CV splits were not
# stratified

fig, ax = plt.subplots(1, 1, figsize=(10, 7))
sns.set_style("darkgrid")
ax = sns.boxplot(x='model', y='test_score', data=df_scores)
ax = sns.swarmplot(x="model", y="test_score", data=df_scores, color=".25")
