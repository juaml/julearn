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
configure_logging(level="DEBUG")

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
model_sepal.add("filter_columns",
                apply_to=["sepal", "petal"],
                keep=["sepal"])
model_sepal.add("svm")

model_petal = PipelineCreator()
model_petal.add("rf")

model = PipelineCreator()

model.add("zscore", apply_to=["sepal", "petal"])
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


model_sepal = PipelineCreator()
model_sepal.add("filter_columns", apply_to=["sepal", "petal"], keep=["sepal"])
model_sepal.add("zscore", apply_to="sepal")
model_sepal.add("svm")

model_petal = PipelineCreator()
model_petal.add("filter_columns", apply_to=["sepal", "petal"], keep=["petal"])
model_petal.add("zscore", apply_to="petal")
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

scores, final_model = run_cross_validation(
    X=X, y=y, X_types=X_types, data=df_iris, model=model,
    return_estimator="final"
)

print(scores["test_score"])
# check whether we really only have to features at the rf
print(final_model[-1].estimators_[-1][-1].feature_importances_)
