"""
Stacking Classification
=======================

This example uses the ``iris`` dataset and performs a complex stacking
classification. We will use two different classifiers, one applied to petal
features and one applied to sepal features. A final logistic regression
classifier will be applied on the predictions of the two classifiers.

.. include:: ../../links.inc
"""
# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
# License: AGPL

from seaborn import load_dataset
from julearn import run_cross_validation
from julearn.pipeline import PipelineCreator
from julearn.utils import configure_logging

###############################################################################
# Set the logging level to info to see extra information.
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

# Define our feature types
X_types = {
    "sepal": ["sepal_length", "sepal_width"],
    "petal": ["petal_length", "petal_width"],
}

# Create the pipeline for the sepal features, by default will apply to "sepal"
model_sepal = PipelineCreator(problem_type="classification", apply_to="sepal")
model_sepal.add("filter_columns", apply_to="*", keep="sepal")
model_sepal.add("zscore")
model_sepal.add("svm")

# Create the pipeline for the petal features, by default will apply to "petal"
model_petal = PipelineCreator(problem_type="classification", apply_to="petal")
model_petal.add("filter_columns", apply_to="*", keep="petal")
model_petal.add("zscore")
model_petal.add("rf")

# Create the stacking model
model = PipelineCreator(problem_type="classification")
model.add(
    "stacking",
    estimators=[[("model_sepal", model_sepal), ("model_petal", model_petal)]],
    apply_to="*",
)

scores = run_cross_validation(
    X=X, y=y, X_types=X_types, data=df_iris, model=model
)

print(scores["test_score"])
