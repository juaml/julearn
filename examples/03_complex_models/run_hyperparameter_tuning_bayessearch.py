"""
Tuning Hyperparameters using Bayesian Search
============================================

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
configure_logging(level="DEBUG")

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
# Following the hyperparamter tuning example, we will now use a Bayesian
# search to find the best hyperparameters for the SVM model.
X = ["frontal", "parietal"]
y = "event"

creator1 = PipelineCreator(problem_type="classification")
creator1.add("zscore")
creator1.add(
    "svm",
    kernel=["linear"], 
    # C=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3],
    C=(1e-6, 1e6, "log-uniform"),
)

# creator2 = PipelineCreator(problem_type="classification")
# creator2.add("zscore")
# creator2.add(
#     "svm",
#     kernel=["rbf", "poly"],
#     C=(1e-6, 1e6, "log-uniform"),
#     gamma=(1e-6, 1e1, "log-uniform"),
# )

search_params = {
    "kind": "bayes",
    "cv": 2,  # to speed up the example
    "n_iter": 10,  # 10 iterations of bayesian search
}

# search_params = {
#     "kind": "grid",
#     "cv": 2,  # to speed up the example
#     # "n_iter": 10,  # 10 iterations of bayesian search
# }

scores, estimator = run_cross_validation(
    X=X,
    y=y,
    data=df_fmri,
    # model=[creator1, creator2],
    model=creator1,
    search_params=search_params,
    return_estimator="final",
)

print(scores["test_score"].mean())


###############################################################################
# It seems that we might have found a better model, but which one is it?
print(estimator.best_params_)
