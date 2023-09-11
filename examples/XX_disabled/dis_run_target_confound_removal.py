"""
Confound Removal (model comparison)
===================================

This example uses the ``iris`` dataset, performs simple binary classification
with and without confound removal using a Random Forest classifier.

"""
# Authors: Shammi More <s.more@fz-juelich.de>
#          Federico Raimondo <f.raimondo@fz-juelich.de>
#          Leonard Sasse <l.sasse@fz-juelich.de>
# License: AGPL

from seaborn import load_dataset

from julearn import run_cross_validation
from julearn.pipeline import PipelineCreator, TargetPipelineCreator
from julearn.utils import configure_logging

###############################################################################
# Set the logging level to info to see extra information.
configure_logging(level="INFO")

###############################################################################
# Load the iris data from seaborn.
df_iris = load_dataset("iris")

###############################################################################
# The dataset has three kind of species. We will keep two to perform a binary
# classification.
df_iris = df_iris[df_iris["species"].isin(["versicolor", "virginica"])]

###############################################################################
# As features, we will use the sepal length, width and petal length and use
# petal width as confound.
X = ["sepal_length", "sepal_width"]
y = "petal_length"
confounds = ["petal_width"]

# In order to tell ``run_cross_validation`` which columns are confounds,
# and which columns are features, we have to define the ``X_types``:
X_types = {"features": X, "confound": confounds}

target_creator = TargetPipelineCreator()
target_creator.add("zscore")
target_creator.add("confound_removal", confounds="confound")

creator = PipelineCreator(problem_type="regression", apply_to="features")
creator.add("zscore", apply_to=["features", "confound"])
creator.add(target_creator, apply_to="target")
creator.add("rf")

scores_cr = run_cross_validation(
    X=X + confounds,
    y=y,
    data=df_iris,
    model=creator,
    cv=5,
    X_types=X_types,
    scoring=["r2"],
    seed=200,
    pos_labels=["virginica"],
)

scores_cr
