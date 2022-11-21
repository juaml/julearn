"""
Simple Binary Classification
============================

This example uses the 'iris' dataset and performs a simple binary
classification using a Support Vector Machine classifier.

.. include:: ../../links.inc
"""
# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#
# License: AGPL
from seaborn import load_dataset
from julearn import run_cross_validation
from julearn.pipeline import PipelineCreator
from julearn.utils import configure_logging

###############################################################################
# Set the logging level to info to see extra information
configure_logging(level="INFO")

###############################################################################
df_iris = load_dataset("iris")

###############################################################################
# The dataset has three kind of species. We will keep two to perform a binary
# classification.
df_iris = df_iris[df_iris["species"].isin(["versicolor", "virginica"])]

###############################################################################
# As features, we will use the sepal length, width and petal length.
# We will try to predict the species.

X = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
y = "species"


model_sepal = PipelineCreator()
model_sepal.add("zscore")
model_sepal.add("svm")

model_petal = PipelineCreator()
model_petal.add("zscore")
model_petal.add("rf")

model = PipelineCreator()
model.add("stacking", estimators=[[
    ('sepal', model_sepal),
    ('petal', model_petal)
]])


X_types = {
    "sepal": ["sepal_length", "sepal_width"],
    "petal": ["petal_length", "petal_width"],
}

scores = run_cross_validation(
    X=X, y=y, X_types=X_types, data=df_iris, model=model
)

print(scores["test_score"])
