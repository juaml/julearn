"""
Inspecting the fold-wise predictions
====================================

This example uses the 'iris' dataset and performs a simple binary
classification using a Support Vector Machine classifier.

We later inspect the predictions of the model for each fold.

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

X = ["sepal_length", "sepal_width", "petal_length"]
y = "species"
X_types = {"continuous": X}

creator = PipelineCreator(problem_type="classification")
creator.add("zscore")
creator.add("svm")

scores = run_cross_validation(
    X=X,
    y=y,
    data=df_iris,
    model=creator,
    return_estimator="cv",
)

print(scores)

###############################################################################
# We can now inspect the predictions of the model for each fold.

cv_predictions = fold_predictions(
    scores=scores,
    cv=cv,
    X=X,
    data=data,
    func=True,
)
