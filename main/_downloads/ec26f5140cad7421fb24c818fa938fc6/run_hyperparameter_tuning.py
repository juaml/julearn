"""
Tuning Hyperparameters
=======================

This example uses the 'fmri' dataset, performs simple binary classification
using a Support Vector Machine classifier and analyse the model.


References
----------
Waskom, M.L., Frank, M.C., Wagner, A.D. (2016). Adaptive engagement of
cognitive control in context-dependent decision-making. Cerebral Cortex.


.. include:: ../../links.inc
"""
# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#
# License: AGPL
import numpy as np
from seaborn import load_dataset

from julearn import run_cross_validation
from julearn.utils import configure_logging

###############################################################################
# Set the logging level to info to see extra information
configure_logging(level='INFO')

###############################################################################
# Set the random seed to always have the same example
np.random.seed(42)


###############################################################################
# Load the dataset
df_fmri = load_dataset('fmri')
print(df_fmri.head())

###############################################################################
# Set the dataframe in the right format
df_fmri = df_fmri.pivot(
    index=['subject', 'timepoint', 'event'],
    columns='region',
    values='signal')

df_fmri = df_fmri.reset_index()
print(df_fmri.head())

###############################################################################
# Lets do a first attempt and use a linear SVM with the default parameters.
model_params = {'svm__kernel': 'linear'}
X = ['frontal', 'parietal']
y = 'event'
scores = run_cross_validation(
    X=X, y=y, data=df_fmri, model='svm', preprocess_X='zscore',
    model_params=model_params)

print(scores['test_score'].mean())

###############################################################################
# The score is not so good. Lets try to see if there is an optimal
# regularization parameter (C) for the linear SVM.
model_params = {
    'svm__kernel': 'linear',
    'svm__C': [0.01, 0.1],
    'cv': 2}  # CV=2 too speed up the example
X = ['frontal', 'parietal']
y = 'event'
scores, estimator = run_cross_validation(
    X=X, y=y, data=df_fmri, model='svm', preprocess_X='zscore',
    model_params=model_params, return_estimator='final')

print(scores['test_score'].mean())

###############################################################################
# This did not change much, lets explore other kernels too.
model_params = {
    'svm__kernel': ['linear', 'rbf', 'poly'],
    'svm__C': [0.01, 0.1],
    'cv': 2}  # CV=2 too speed up the example
X = ['frontal', 'parietal']
y = 'event'
scores, estimator = run_cross_validation(
    X=X, y=y, data=df_fmri, model='svm', preprocess_X='zscore',
    model_params=model_params, return_estimator='final')

print(scores['test_score'].mean())
###############################################################################
# It seems that we might have found a better model, but which one is it?
print(estimator.best_params_)

###############################################################################
# Now that we know that a RBF kernel is better, lest test different *gamma*
# parameters.
model_params = {
    'svm__kernel': 'rbf',
    'svm__C': [0.01, 0.1],
    'svm__gamma': [1e-2, 1e-3],
    'cv': 2}  # CV=2 too speed up the example
X = ['frontal', 'parietal']
y = 'event'
scores, estimator = run_cross_validation(
    X=X, y=y, data=df_fmri, model='svm', preprocess_X='zscore',
    model_params=model_params, return_estimator='final')

print(scores['test_score'].mean())
print(estimator.best_params_)

###############################################################################
# It seems that without tuning the gamma parameter we had a better accuracy.
# Let's add the default value and see what happens.
model_params = {
    'svm__kernel': 'rbf',
    'svm__C': [0.01, 0.1],
    'svm__gamma': [1e-2, 1e-3, 'scale'],
    'cv': 2}  # CV=2 too speed up the example
X = ['frontal', 'parietal']
y = 'event'
scores, estimator = run_cross_validation(
    X=X, y=y, data=df_fmri, model='svm', preprocess_X='zscore',
    model_params=model_params, return_estimator='final')

print(scores['test_score'].mean())
print(estimator.best_params_)

###############################################################################
# So what was the best ``gamma`` in the end?
print(estimator.best_estimator_['svm']._gamma)
