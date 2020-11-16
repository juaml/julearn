"""
Confound Removal in Advanced Settings
=====================================

In most cases confound removal is a simple operation.
You regress out the confound from the features and only continue working with
these new deconfounded features. This example will not focus on these
typical cases. Instead, this will show how to keep the confounds and use
them as additional features.

.. include:: ../../links.inc
"""
# Authors: Sami Hamdan <s.hamdan@fz-juelich.de>
#
# License: AGPL
from sklearn.datasets import load_diabetes  # to load data
from julearn.pipeline import create_extended_pipeline
from julearn.transformers import (get_transformer,
                                  DataFrameConfoundRemover,
                                  ChangeColumnTypes)
from julearn.estimators import get_model


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
# In the following we will explore different settings of confound removal
# using Julearns pipeline functionalities.
#
# .. note::
#  Everything, shown here is also possible in Julearns `run_cross_validation`
#  function.
#
# Confound Removal Typical Use Case
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Here, we first deconfound the features.
# The confound is automatically dropped by the `.DataFrameConfoundRemover`.
# Afterwards, we will transform our features with a pca and run
# a lienar regression
#
typical_pipe = create_extended_pipeline(
    preprocess_steps_features=[
        ('remove_confound', DataFrameConfoundRemover(), 'subset', 'all'),
        ('pca', *get_transformer('pca'))
    ],
    preprocess_steps_confounds=None,
    preprocess_transformer_target=None,
    model=('lr', get_model('linreg', 'regression')),
    confounds=['sex'],
    categorical_features=None,
)
typical_pipe.fit(df_features, target)

###############################################################################
# We can use the `preprocess` method of the `.ExtendedDataFramePipeline`
# to inspect the transformations/preprocessing steps of our pipeline.
# By providing a step name to the `until` argument of the
# `preprocess` method we return the transformed X and y including to this step.
# This output always includes the transformed
# X and tranformed y as a tuple.
X_deconfounded, _ = typical_pipe.preprocess(
    df_features, target, until='remove_confound')
print(X_deconfounded.head())

###############################################################################
# As you can see the confound `sex` was dropped and only the deconfounded
# features are used in the following pca.
# But what if you want to keep the confound in the feature space.
#
# For example, let's assume that you want to do a pca on the deconfounded
# features, but still want to keep the confound as a feature.
# This would mean that the following pca would also use the confound
# as a feature.
# Let us have a closer look to the cofound remover in order to understand
# how we could achieve such a task:
#
# .. autoclass:: julearn.transformers.DataFrameConfoundRemover

###############################################################################
# As you can see above we can set the `keep_confouns` argument to True,
# if we want to keep the confounds.
# Here is an example of how this can look like:

keep_confound_pipe = create_extended_pipeline(
    preprocess_steps_features=[
        ('remove_confound', DataFrameConfoundRemover(
            keep_confounds=True), 'subset', 'all'),
        ('pca', *get_transformer('pca'))
    ],
    preprocess_steps_confounds=None,
    preprocess_transformer_target=None,
    model=('lr', get_model('linreg', 'regression')),
    confounds=['sex'],
    categorical_features=None,
)
keep_confound_pipe.fit(df_features, target)

###############################################################################
# As you can see this will keep the confound
X_deconfounded, _ = keep_confound_pipe.preprocess(
    df_features, target, until='remove_confound')
print(X_deconfounded.head())

###############################################################################
# Even after the pca the confound will still be present
# This is the case because by default transformers only transform continuous
# features (including features without a specified type)
# and the confound is of type confound.
X_transformed, _ = keep_confound_pipe.preprocess(df_features, target)
print(X_transformed.head())

# This means that the resulting Linear Regression will use the deconfounded
# features together with the confound to predict the target.

###############################################################################
# Lastly, you can also use the confound as a normal feature after confound
# removal. To do so you can either add the confound(s) to the
# transformed_columns of the each following transformers
# which return the same columns or you can use the
# `.ChangeColumnTypes` to change the returned confounds
# to a continuous variable like this:
confound_as_feature_pipe = create_extended_pipeline(
    preprocess_steps_features=[
        ('remove_confound', DataFrameConfoundRemover(
            keep_confounds=True), 'subset', 'all'),
        ('change_column_types', ChangeColumnTypes('.*confound', 'continuous'),
         'from_transformer', 'all'),
        ('pca', *get_transformer('pca'))
    ],
    preprocess_steps_confounds=None,
    preprocess_transformer_target=None,
    model=('lr', get_model('linreg', 'regression')),
    confounds=['sex'],
    categorical_features=None,
)
confound_as_feature_pipe.fit(df_features, target)

###############################################################################
# As you can see this will keep the confound and
# change its type to a continuous variable.
X_deconfounded, _ = confound_as_feature_pipe.preprocess(
    df_features, target, until='change_column_types',
    return_trans_column_type=True)
print(X_deconfounded.head())

###############################################################################
# Because the confound is treated as a normal continuous feature
# after removal it will be transformed in the pca as well
X_transformed, _ = confound_as_feature_pipe.preprocess(df_features, target)
print(X_transformed.head())
