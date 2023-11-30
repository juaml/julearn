"""
Tuning Multiple Hyperparameters Grids
=====================================

This example uses the ``fmri`` dataset, performs simple binary classification
using a Support Vector Machine classifier while tuning multiple hyperparameters
grids at the same time.

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
# Lets do a first attempt and use a linear SVM with the default parameters.

X = ["frontal", "parietal"]
y = "event"

creator = PipelineCreator(problem_type="classification")
creator.add("zscore")
creator.add("svm", kernel="linear")

scores = run_cross_validation(X=X, y=y, data=df_fmri, model=creator)

print(scores["test_score"].mean())

###############################################################################
# Now let's tune a bit this SVM. We will use a grid search to tune the
# regularization parameter ``C`` and the kernel. We will also tune the ``gamma``.
# But since the ``gamma`` is only used for the rbf kernel, we will use a
# different grid for the ``"rbf"`` kernel.
#
# To specify two different sets of parameters for the same step, we can
# explicitly specify the name of the step. This is done by passing the
# ``name`` parameter to the ``add`` method.
creator = PipelineCreator(problem_type="classification")
creator.add("zscore")
creator.add("svm", kernel="linear", C=[0.01, 0.1], name="svm")
creator.add(
    "svm",
    kernel="rbf",
    C=[0.01, 0.1],
    gamma=["scale", "auto", 1e-2, 1e-3],
    name="svm",
)

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
# It seems that we might have found a better model, but which one is it?
print(estimator.best_params_)
print(estimator.best_estimator_["svm"]._gamma)
