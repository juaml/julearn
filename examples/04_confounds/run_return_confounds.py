"""
Return Confounds in Confound Removal
====================================

In most cases confound removal is a simple operation.
You regress out the confound from the features and only continue working with
these new confound removed features. This is also the default setting for
``julearn``'s ``remove_confound`` step. But sometimes you want to work with the
confound even after removing it from the features. In this example, we
will discuss the options you have.

.. include:: ../../links.inc
"""
# Authors: Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL

from sklearn.datasets import load_diabetes  # to load data
from julearn.pipeline import PipelineCreator
from julearn import run_cross_validation
from julearn.inspect import preprocess

# Load in the data
df_features, target = load_diabetes(return_X_y=True, as_frame=True)

###############################################################################
# First, we can have a look at our features.
# You can see it includes Age, BMI, average blood pressure (bp) and 6 other
# measures from s1 to s6. Furthermore, it includes sex which will be considered
# as a confound in this example.
print("Features: ", df_features.head())

###############################################################################
# Second, we can have a look at the target.
print("Target: ", target.describe())

###############################################################################
# Now, we can put both into one DataFrame:
data = df_features.copy()
data["target"] = target

###############################################################################
# In the following we will explore different settings of confound removal
# using ``julearn``'s pipeline functionalities.
#
# Confound Removal Typical Use Case
# ---------------------------------
# Here, we want to deconfound the features and not include the confound as a
# feature into our last model. We will use the ``remove_confound`` step for this.
# Then we will use the ``pca`` step to reduce the dimensionality of the features.
# Finally, we will fit a linear regression model.

creator = PipelineCreator(problem_type="regression", apply_to="continuous")
creator.add("confound_removal")
creator.add("pca")
creator.add("linreg")

###############################################################################
# Now we need to set the ``X_types`` argument of the ``run_cross_validation``
# function. This argument is a dictionary that maps the names of the different
# types of X to the features that belong to this type. In this example, we
# have two types of features: `continuous` and `confound`. The `continuous`
# features are the features that we want to deconfound and the `confound`
# features are the features that we want to remove from the `continuous`.

feature_names = list(df_features.drop(columns="sex").columns)
X_types = {"continuous": feature_names, "confound": "sex"}

X = feature_names + ["sex"]

###############################################################################
# Now we can run the cross validation and get the scores.
scores, model = run_cross_validation(
    X=X,
    y="target",
    X_types=X_types,
    data=data,
    model=creator,
    return_estimator="final",
)

###############################################################################
# We can use the ``preprocess`` method of the ``inspect`` module to inspect the
# transformations steps of the returned estimator.
# By providing a step name to the ``until`` argument of the
# ``preprocess`` method we return the transformed X and y up to
# the provided step (inclusive).
df_deconfounded = preprocess(model, X=X, data=data, until="confound_removal")
df_deconfounded.head()

###############################################################################
# As you can see the confound ``sex`` was dropped and only the confound removed
# features are used in the following PCA.
#
# But what if you want to keep the confound after removal for
# other transformations?
#
# For example, let's assume that you want to do a PCA on the confound removed
# feature, but want to keep the confound for the actual modelling step.
# Let us have a closer look to the confound remover in order to understand
# how we could achieve such a task:
#
# .. autoclass:: julearn.transformers.confound_remover.ConfoundRemover
#    :noindex:
#    :exclude-members: transform, get_support, get_feature_names_out,
#                      filter_columns, fit, fit_transform, get_apply_to,
#                      get_needed_types, get_params, set_output, set_params

###############################################################################
# In this example, we will set the ``keep_confounds`` argument to True.
# This will keep the confounds after confound removal.

creator = PipelineCreator(problem_type="regression", apply_to="continuous")
creator.add("confound_removal", keep_confounds=True)
creator.add("pca")
creator.add("linreg")

###############################################################################
# Now we can run the cross validation and get the scores.
scores, model = run_cross_validation(
    X=X,
    y="target",
    X_types=X_types,
    data=data,
    model=creator,
    return_estimator="final",
)

###############################################################################
# As you can see this kept the confound variable ``sex`` in the data.
df_deconfounded = preprocess(model, X=X, data=data, until="confound_removal")
df_deconfounded.head()

###############################################################################
# Even after the PCA, the confound will still be present.
# This is the case because by default transformers only transform continuous
# features (including features without a specified type) and ignore confounds
# and categorical variables.
df_transformed = preprocess(model, X=X, data=data)
df_transformed.head()

###############################################################################
# This means that the resulting Linear Regression can use the deconfounded
# features together with the confound to predict the target. However, in the
# pipeline creator, the model is only applied to the continuous features.
# This means that the confound is not used in the model.
# Here we can see that the model is using 9 features.

print(len(model.steps[-1][1].model.coef_))

###############################################################################
# Lastly, you can also use the confound as a normal feature after confound
# removal.
creator = PipelineCreator(problem_type="regression", apply_to="continuous")
creator.add("confound_removal", keep_confounds=True)
creator.add("pca")
creator.add("linreg", apply_to="*")

scores, model = run_cross_validation(
    X=X,
    y="target",
    X_types=X_types,
    data=data,
    model=creator,
    return_estimator="final",
)
scores

###############################################################################
# As you can see the confound is now used in the linear regression model.
# This is the case because we set the ``apply_to`` argument of the ``linreg``
# step to ``*``. This means that the step will be applied to all features
# (including confounds and categorical variables).
# Here we can see that the model is using 10 features (9 deconfounded features
# and the confound).
print(len(model.steps[-1][1].model.coef_))
