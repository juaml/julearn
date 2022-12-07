"""
Return Confounds in Confound Removal
====================================

In most cases confound removal is a simple operation.
You regress out the confound from the features and only continue working with
these new confound removed features. This is also the default setting for
julearn's `remove_confound` step. But sometimes you want to work with the
confound even after removing it from the features. In this example, we
will discuss the options you have.

.. include:: ../../links.inc
"""
# Authors: Sami Hamdan <s.hamdan@fz-juelich.de>
#
# License: AGPL
from sklearn.datasets import load_diabetes  # to load data
from julearn.transformers import ChangeColumnTypes
from julearn import run_cross_validation
import warnings

# load in the data
df_features, target = load_diabetes(return_X_y=True, as_frame=True)


###############################################################################
# First, we can have a look at our features.
# You can see it includes
# Age, BMI, average blood pressure (bp) and 6 other measures from s1 to s6
# Furthermore, it includes sex which will be considered as a confound in
# this example.
#
print('Features: ', df_features.head())

###############################################################################
# Second, we can have a look at the target
print('Target: ', target.describe())

###############################################################################
# Now, we can put both into one DataFrame:
data = df_features.copy()
data['target'] = target

###############################################################################
# In the following we will explore different settings of confound removal
# using Julearns pipeline functionalities.
#
# Confound Removal Typical Use Case
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Here, we want to deconfound the features and not include the confound as a
# feature into our last model.
# Afterwards, we will transform our features with a pca and run
# a linear regression.
#
feature_names = list(df_features.drop(columns='sex').columns)
with warnings.catch_warnings():
    warnings.simplefilter("ignore", lineno=443)
    scores, model = run_cross_validation(
        X=feature_names, y='target', data=data,
        confounds='sex', model='linreg', problem_type='regression',
        preprocess_X=['remove_confound', 'pca'],
        return_estimator='final')

###############################################################################
# We can use the `preprocess` method of the `.ExtendedDataFramePipeline`
# to inspect the transformations/preprocessing steps of the returned estimator.
# By providing a step name to the `until` argument of the
# `preprocess` method we return the transformed X and y up to
# the provided step (inclusive).
# This output consists of a tuple containing the transformed X and y,
X_deconfounded, _ = model.preprocess(
    df_features, target, until='remove_confound')
print(X_deconfounded.head())

# As you can see the confound `sex` was dropped
# and only the confound removed features are used in the following pca.
# But what if you want to keep the confound after removal for
# other transformations.
#
# For example, let's assume that you want to do a pca on the confound removed
# feature, but want to keep the confound for the actual modelling step.
# Let us have a closer look to the confound remover in order to understand
# how we could achieve such a task:
#
# .. autoclass:: julearn.transformers.DataFrameConfoundRemover

###############################################################################
# Above, you can see that we can set the `keep_confounds` argument to True.
# This will keep the confounds after confound removal.
# Here, is an example of how this can look like:

scores, model = run_cross_validation(
    X=feature_names, y='target', data=data,
    confounds='sex', model='linreg', problem_type='regression',
    preprocess_X=['remove_confound', 'pca'],
    model_params=dict(remove_confound__keep_confounds=True),
    return_estimator='final')

###############################################################################
# As you can see this will keep the confound
X_deconfounded, _ = model.preprocess(
    df_features, target, until='remove_confound')
print(X_deconfounded.head())

###############################################################################
# Even after the pca the confound will still be present.
# This is the case because by default transformers only transform continuous
# features (including features without a specified type)
# and ignore confounds and categorical variables.
X_transformed, _ = model.preprocess(df_features, target)
print(X_transformed.head())

# This means that the resulting Linear Regression will use the deconfounded
# features together with the confound to predict the target.

###############################################################################
# Lastly, you can also use the confound as a normal feature after confound
# removal. To do so you can either add the confound(s) to the
# which return the same columns or you can use the
# `.ChangeColumnTypes` to change the returned confounds
# to a continuous variable like this:
scores, model = run_cross_validation(
    X=feature_names, y='target', data=data,
    confounds='sex', model='linreg', problem_type='regression',
    preprocess_X=['remove_confound',
                  ChangeColumnTypes('.*confound', 'continuous'),
                  'pca'],
    preprocess_confounds='zscore',
    model_params=dict(remove_confound__keep_confounds=True),
    return_estimator='final'
)


###############################################################################
# As you can see this will keep the confound and
# change its type to a continuous variable.
X_deconfounded, _ = model.preprocess(
    df_features, target, until='changecolumntypes',
    return_trans_column_type=True)
print(X_deconfounded.head())

###############################################################################
# Because the confound is treated as a normal continuous feature
# after removal it will be transformed in the pca as well
X_transformed, _ = model.preprocess(df_features, target)
print(X_transformed.head())
