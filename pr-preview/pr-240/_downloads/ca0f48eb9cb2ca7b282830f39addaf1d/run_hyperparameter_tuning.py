"""
Tuning Hyperparameters
=======================

This example uses the ``fmri`` dataset, performs simple binary classification
using a Support Vector Machine classifier and analyze the model.

References
----------

  Waskom, M.L., Frank, M.C., Wagner, A.D. (2016). Adaptive engagement of
  cognitive control in context-dependent decision-making. Cerebral Cortex.

.. include:: ../../links.inc
"""
# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
# License: AGPL

import numpy as np
from seaborn import load_dataset

from julearn import run_cross_validation
from julearn.utils import configure_logging
from julearn.pipeline import PipelineCreator

###############################################################################
# Set the logging level to info to see extra information.
configure_logging(level="INFO")

###############################################################################
# Set the random seed to always have the same example.
np.random.seed(42)

###############################################################################
# Load the dataset.
df_fmri = load_dataset("fmri")
df_fmri.head()

###############################################################################
# Set the dataframe in the right format.
df_fmri = df_fmri.pivot(
    index=["subject", "timepoint", "event"], columns="region", values="signal"
)

df_fmri = df_fmri.reset_index()
df_fmri.head()

###############################################################################
# Let's do a first attempt and use a linear SVM with the default parameters.
X = ["frontal", "parietal"]
y = "event"

creator = PipelineCreator(problem_type="classification")
creator.add("zscore")
creator.add("svm", kernel="linear")

scores = run_cross_validation(X=X, y=y, data=df_fmri, model=creator)

print(scores["test_score"].mean())

###############################################################################
# The score is not so good. Let's try to see if there is an optimal
# regularization parameter (C) for the linear SVM.
# We will use a grid search to find the best ``C``.

creator = PipelineCreator(problem_type="classification")
creator.add("zscore")
creator.add("svm", kernel="linear", C=[0.01, 0.1])

search_params = {
    "kind": "grid",
    "cv": 2,  # to speed up the example
}

scores, estimator = run_cross_validation(
    X=X,
    y=y,
    data=df_fmri,
    model=creator,
    search_params=search_params,
    return_estimator="final",
)

print(scores["test_score"].mean())

###############################################################################
# This did not change much, lets explore other kernels too.

creator = PipelineCreator(problem_type="classification")
creator.add("zscore")
creator.add("svm", kernel=["linear", "rbf", "poly"], C=[0.01, 0.1])

scores, estimator = run_cross_validation(
    X=X,
    y=y,
    data=df_fmri,
    model=creator,
    search_params=search_params,
    return_estimator="final",
)

print(scores["test_score"].mean())
###############################################################################
# It seems that we might have found a better model, but which one is it?
print(estimator.best_params_)

###############################################################################
# Now that we know that a RBF kernel is better, lest test different *gamma*
# parameters.

creator = PipelineCreator(problem_type="classification")
creator.add("zscore")
creator.add("svm", kernel="rbf", C=[0.01, 0.1], gamma=[1e-2, 1e-3])

scores, estimator = run_cross_validation(
    X=X,
    y=y,
    data=df_fmri,
    model=creator,
    search_params=search_params,
    return_estimator="final",
)

print(scores["test_score"].mean())
print(estimator.best_params_)

###############################################################################
# It seems that without tuning the gamma parameter we had a better accuracy.
# Let's add the default value and see what happens.

creator = PipelineCreator(problem_type="classification")
creator.add("zscore")
creator.add("svm", kernel="rbf", C=[0.01, 0.1], gamma=[1e-2, 1e-3, "scale"])
X = ["frontal", "parietal"]
y = "event"

search_params = {"cv": 2}

scores, estimator = run_cross_validation(
    X=X,
    y=y,
    data=df_fmri,
    model=creator,
    return_estimator="final",
    search_params=search_params,
)

print(scores["test_score"].mean())
print(estimator.best_params_)

###############################################################################
print(estimator.best_estimator_["svm"]._gamma)
