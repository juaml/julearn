"""
Preprocessing with variance threshold, zscore and PCA
=====================================================

This example uses the ``make_regression`` function to create a simple dataset,
performs a simple regression after the preprocessing of the features
including removal of low variance features, feature normalization for only
two features using zscore and feature reduction using PCA.
We will check the features after each preprocessing step.
"""
# Authors: Shammi More <s.more@fz-juelich.de>
#          Leonard Sasse <l.sasse@fz-juelich.de>
# License: AGPL

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import make_regression

from julearn import run_cross_validation
from julearn.inspect import preprocess
from julearn.pipeline import PipelineCreator
from julearn.utils import configure_logging

###############################################################################
# Set the logging level to info to see extra information.
configure_logging(level="INFO")

###############################################################################
# Create a dataset using ``sklearn`` ``make_regression``.
df = pd.DataFrame()
X, y = [f"Feature {x}" for x in range(1, 5)], "y"
df[X], df[y] = make_regression(
    n_samples=100, n_features=4, n_informative=3, noise=0.3, random_state=0
)

# We only want to zscore the first two features, so let's get their names.
first_two = X[:2]

# We can define a dictionary, in which the 'key' defines the names of our
# different 'types' of 'X'. The 'value' determine, which features belong to
# this type.
X_types = {"X_to_zscore": first_two}

###############################################################################
# Let's look at the summary statistics of the raw features.
print("Summary Statistics of the raw features : \n", df.describe())

###############################################################################
# We will preprocess all features using variance thresholding.
# We will only zscore the first two features, and then perform PCA using all
# features. We will zscore the target and then train a random forest model.
# Since we use the PipelineCreator object we have to explicitly declare which
# `X_types` each preprocessing step should be applied to. If we do not declare
# the type in the ``add`` method using the ``apply_to`` keyword argument,
# the step will default to ``"continuous"`` or to another type that can be
# declared in the constructor of the ``PipelineCreator``.
# To transform the target we could set ``apply_to="target"``, which is a special
# type, that cannot be user-defined. Please note also that if a step is added
# to transform the target, you also have to explicitly add the model that is
# to be used in the regression to the ``PipelineCreator``.

# Define model parameters and preprocessing steps first
# The hyperparameters for each step can be added as a keyword argument and
# should be either one parameter or an iterable of multiple parameters for a
# search.

# Setting the threshold for variance to 0.15, number of PCA components to 2
# and number of trees for random forest to 200.

# By setting "apply_to=*", we can apply the preprocessing step to all features.
pipeline_creator = PipelineCreator(problem_type="regression")

pipeline_creator.add("select_variance", apply_to="*", threshold=0.15)
pipeline_creator.add("zscore", apply_to="X_to_zscore")
pipeline_creator.add("pca", apply_to="*", n_components=2)
pipeline_creator.add("rf", apply_to="*", n_estimators=200)

# Because we have already added the model to the pipeline creator, we can
# simply drop in the ``pipeline_creator`` as a model. If we did not add a model
# here, we could add the ``pipeline_creator`` using the keyword argument
# ``preprocess`` and hand over a model separately.

scores, model = run_cross_validation(
    X=X,
    y=y,
    X_types=X_types,
    data=df,
    model=pipeline_creator,
    scoring=["r2", "neg_mean_absolute_error"],
    return_estimator="final",
    seed=200,
)

# We can use the final estimator to inspect the transformed features at a
# specific step of the pipeline. Since the PCA was the last step added to the
# pipeline, we can simply get the model up to this step by indexing as follows:

X_after_pca = model[:-1].transform(df[X])

print("X after PCA:")
print("=" * 79)
print(X_after_pca)

# We can select pipelines up to earlier steps by indexing previous elements
# in the final estimator. For example, to inspect features after the zscoring
# step:

X_after_zscore = model[:-2].transform(df[X])
print("X after zscore:")
print("=" * 79)
print(X_after_zscore)

# However, to make this less confusing you can also simply use the high-level
# function ``preprocess`` to explicitly refer to a pipeline step by name:

X_after_pca = preprocess(model, X=X, data=df, until="pca")
X_after_zscore = preprocess(model, X=X, data=df, until="zscore")

# Let's plot scatter plots for raw features and the PCA components.
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
sns.scatterplot(x=X[0], y=X[1], data=df, ax=axes[0])
axes[0].set_title("Raw features")
sns.scatterplot(x="pca__pca0", y="pca__pca1", data=X_after_pca, ax=axes[1])
axes[1].set_title("PCA components")

###############################################################################
# Let's look at the summary statistics of the zscored features. We see here
# that the mean of all the features is zero and standard deviation is one.
print(
    "Summary Statistics of the zscored features : \n",
    X_after_zscore.describe(),
)
