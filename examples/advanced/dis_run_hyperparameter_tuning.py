"""
Tuning Hyperparameters
======================

This example uses the 'fmri' dataset, performs simple binary classification
using a Support Vector Machine classifier and analyses the model.


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

from julearn.pipeline import PipelineCreator
from julearn import run_cross_validation
from julearn.utils import configure_logging

###############################################################################
# Set the logging level to info to see extra information
configure_logging(level="DEBUG")

###############################################################################
# Set the random seed to always have the same example
np.random.seed(42)


###############################################################################
# Load the dataset
df_fmri = load_dataset("fmri")
print(df_fmri.head())

###############################################################################
# Set the dataframe in the right format
df_fmri = df_fmri.pivot(
    index=["subject", "timepoint", "event"], columns="region", values="signal"
)

df_fmri = df_fmri.reset_index()
print(df_fmri.head())

###############################################################################
# Let's do a first attempt and use a linear SVM with the default parameters.
X = ['frontal', 'parietal']
y = 'event'

# Create a pipeline with z-score preprocessing and SVM
creator = (PipelineCreator()
            .add("zscore", apply_to="*")  # all columns considered continous
            .add("svm", problem_type="classification", kernel="linear"))

scores = run_cross_validation(
    X=X, y=y, data=df_fmri, model=creator)

print(f"Test score: {scores['test_score'].mean()}")

###############################################################################
# The score is not so good. Lets try to see if there is an optimal
# regularization parameter (C) for the linear SVM.
X = ["frontal", "parietal"]
y = "event"

# Create a pipeline with z-score preprocessing and SVM
creator = (PipelineCreator()
            .add("zscore", apply_to="*")  # all columns considered continous
            .add("svm", problem_type="classification",
                 kernel="linear", C=[0.01, 0.1]))  # TODO HPT bug -> wait for Sami to be fixed
model_params = {
    "cv": 2,  # speed up the example
}

scores, estimator = run_cross_validation(
    X=X,
    y=y,
    data=df_fmri,
    model=creator,
    model_params=model_params,
    return_estimator="final",
)

print(f"Test score: {scores['test_score'].mean()}")

###############################################################################
# This did not change much, lets explore other kernels too.
X = ["frontal", "parietal"]
y = "event"

# Create a pipeline with z-score preprocessing and SVM
creator = (PipelineCreator()
            .add("zscore", apply_to="*")  # all columns considered continous
            .add("svm", problem_type="classification",
                 kernel=["linear", "rbf", "poly"],
                 C=[0.01, 0.1]))  # TODO HPT bug -> wait for Sami to be fixed
model_params = {
    "cv": 2,  # speed up the example
}

scores, estimator = run_cross_validation(
    X=X,
    y=y,
    data=df_fmri,
    model=creator,
    model_params=model_params,
    return_estimator="final",
)

print(f"Test score: {scores['test_score'].mean()}")
###############################################################################
# It seems that we might have found a better model, but which one is it?
print(f"Best parameters of final estimator {estimator.best_params_}")  # TODO double check after HPT works

###############################################################################
# Now that we know that a RBF kernel is better, lest test different *gamma*
# parameters.
X = ["frontal", "parietal"]
y = "event"

# Create a pipeline with z-score preprocessing and SVM
creator = (PipelineCreator()
            .add("zscore", apply_to="*")  # all columns considered continous
            .add("svm", problem_type="classification",
                 kernel=["rbf"],
                 C=[0.01, 0.1],
                 gamma=[1e-2, 1e-3]))  # TODO HPT bug -> wait for Sami to be fixed
model_params = {
    "cv": 2,  # speed up the example
}

scores, estimator = run_cross_validation(
    X=X,
    y=y,
    data=df_fmri,
    model=creator,
    model_params=model_params,
    return_estimator="final",
)

print(f"Test score: {scores['test_score'].mean()}")
print(f"Best parameters of final estimator {estimator.best_params_}")  # TODO double check after HPT works

###############################################################################
# It seems that without tuning the gamma parameter we had a better accuracy.
# Let's add the default value and see what happens.
X = ["frontal", "parietal"]
y = "event"

# Create a pipeline with z-score preprocessing and SVM
creator = (PipelineCreator()
            .add("zscore", apply_to="*")  # all columns considered continous
            .add("svm", problem_type="classification",
                 kernel=["rbf"],
                 C=[0.01, 0.1],
                 gamma=[1e-2, 1e-3, "scale"]))  # TODO HPT bug -> wait for Sami to be fixed
model_params = {
    "cv": 2,  # speed up the example
}

scores, estimator = run_cross_validation(
    X=X,
    y=y,
    data=df_fmri,
    model=creator,
    model_params=model_params,
    return_estimator="final",
)

print(f"Test score: {scores['test_score'].mean()}")
print(f"Best parameters of final estimator {estimator.best_params_}")  # TODO double check after HPT works

###############################################################################
# So what was the best ``gamma`` in the end?
print(estimator.best_estimator_["svm"]._gamma)   # TODO double check after HPT works
