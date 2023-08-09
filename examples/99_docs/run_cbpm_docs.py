"""
Connectome-based Predictive Modeling (CBPM)
===========================================

Applications of machine learning in neuroimaging research typically have to
deal with high-dimensional input features. This can be quite problematic due to
the curse of dimensionality, especially when sample sizes are low at the same
time. Recently, connectome-based predictive modeling (CBPM) has been proposed
as an approach to deal with this problem [#1]_ in regression. This approach
has been used to predict fluid intelligence [#2]_ as well sustained attention
[#3]_ based on brain functional connectivity.

In a nutshell, CBPM consists of:

1. Feature selection
2. Feature aggregation
3. Model building

In CBPM, features are selected if their correlation to the target is
significant according to some specified significance threshold alpha. These
selected features are then summarized according to an aggregation function and
subsequently used to fit a machine learning model. Most commonly in this
approach a linear model is used for this, but in principle it could be any
other machine learning model.

CBPM in ``julearn``
-------------------

``julearn`` implements a simple, ``scikit-learn`` compatible transformer
("cbpm"), that performs the first two parts of this approach, i.e., the feature
selection and feature aggregation. Leveraging ``julearn``'s ``PipelineCreator``,
one can therefore easily apply the ``"cbpm"`` transformer as a preprocessing
step, and then apply any ``scikit-learn``-compatible estimator for the model
building part.

For example, to build a simple CBPM workflow, you can create a pipeline and
run a cross-validation as follows:
"""
# Authors: Leonard Sasse <l.sasse@fz-juelich.de>
# License: AGPL

from julearn import run_cross_validation
from julearn.pipeline import PipelineCreator

from sklearn.datasets import make_regression
import pandas as pd

# Prepare data
X, y = make_regression(n_features=20, n_samples=200)

# Make dataframe
X_names = [f"feature_{x}" for x in range(1, 21)]
data = pd.DataFrame(X)
data.columns = X_names
data["target"] = y

# Prepare a pipeline creator
cbpm_pipeline_creator = PipelineCreator(problem_type="regression")
cbpm_pipeline_creator.add("cbpm")
cbpm_pipeline_creator.add("linreg")

# Cross-validate the cbpm pipeline
scores, final_model = run_cross_validation(
    data=data,
    X=X_names,
    y="target",
    model=cbpm_pipeline_creator,
    return_estimator="all",
)

###############################################################################
# By default the ``"cbpm"`` transformer will perform feature selection using the
# Pearson correlation between each feature and the target, and select the
# features for which the p-value of the correlation falls below the default
# significance threshold of 0.01. It will then group the features into
# negatively and positively correlated features, and sum up the features within
# each of these groups using :func:`numpy.sum`. That is, the linear model in
# this case is fitted on two features:
#
# 1. Sum of features that are positively correlated to the target
# 2. Sum of features that are negatively correlated to the target
#
# The pipeline creator also allows easily customising these parameters of the
# ``"cbpm"`` transformer according to your needs. For example, to use a different
# significance threshold during feature selection one may set the
# ``significance_threshold`` keyword to increase it to 0.05 as follows:

# Prepare a pipeline creator
cbpm_pipeline_creator = PipelineCreator(problem_type="regression")
cbpm_pipeline_creator.add("cbpm", significance_threshold=0.05)
cbpm_pipeline_creator.add("linreg")

print(cbpm_pipeline_creator)

###############################################################################
# ``julearn`` also allows this to be tuned as a hyperparameter in a nested
# cross-validation. Simply hand over an iterable of values:

# Prepare a pipeline creator
cbpm_pipeline_creator = PipelineCreator(problem_type="regression")
cbpm_pipeline_creator.add("cbpm", significance_threshold=[0.01, 0.05])
cbpm_pipeline_creator.add("linreg")

print(cbpm_pipeline_creator)

###############################################################################
# In addition, it may be noteworthy, that you can customize the correlation
# method, the aggregation method, as well as the sign (``"pos"``, ``"neg"``,
# or ``"posneg"``) of the feature-target correlations that should be selected.
# For example, a pipeline that specifies each of these parameters may look as
# follows:

import numpy as np
from scipy.stats import spearmanr

# Prepare a pipeline creator
cbpm_pipeline_creator = PipelineCreator(problem_type="regression")
cbpm_pipeline_creator.add(
    "cbpm",
    significance_threshold=0.05,
    corr_method=spearmanr,
    agg_method=np.average,
    corr_sign="pos",
)
cbpm_pipeline_creator.add("linreg")

print(cbpm_pipeline_creator)

###############################################################################
# As you may have guessed, this pipeline will use a Spearman correlation and a
# significance level of 0.05 for feature selection. It will only select
# features that are positively correlated to the target and aggregate them
# using the :func:`numpy.average` aggregation function.
#
# .. topic:: References:
#
#   .. [#1] Shen, Xilin, et al., `"Using connectome-based predictive modeling \
#      to predict individual behavior from brain connectivity" \
#      <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5526681/>`_, Nat Protoc.
#      2017 Mar; 12(3): 506–518.
#
#   .. [#2] Finn, Emily S., et al., `"Functional connectome fingerprinting: \
#      identifying individuals using patterns of brain connectivity" \
#      <https://pubmed.ncbi.nlm.nih.gov/26457551/>`_, Nat Neurosci. 2015
#      Nov;18(11):1664-71.
#
#   .. [#3] Rosenberg, Monica D., et al., `"A neuromarker of sustained \
#      attention from whole-brain functional connectivity" \
#      <https://pubmed.ncbi.nlm.nih.gov/26595653/>`_, Nat Neurosci. 2016 Jan;
#      19(1):165-71.
