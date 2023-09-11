"""
Parallelize ``julearn``
=======================

In this example we will parallelize outer cross-validation
and/or inner cross-validation for hyperparameter search.

.. include:: ../../links.inc
"""
# Authors: Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL

from seaborn import load_dataset
from julearn import run_cross_validation

###############################################################################
# Prepare some simple standard input.
df_iris = load_dataset("iris")
df_iris = df_iris[df_iris["species"].isin(["versicolor", "virginica"])]

X = ["sepal_length", "sepal_width", "petal_length"]
y = "species"

###############################################################################
# Run without any parallelization.
model_params = {
    "svm__C": [1, 2, 3],
}

scores = run_cross_validation(
    X=X,
    y=y,
    data=df_iris,
    model="svm",
    problem_type="classification",
    model_params=model_params,
)

###############################################################################
# To add parallelization to the outer cross-validation we will add the ``n_jobs``
# argument to ``run_cross_validation``. We can use ``verbose > 0`` to get more
# information about the parallelization done. Here, we'll set the parallel jobs
# to 2.
scores = run_cross_validation(
    X=X,
    y=y,
    data=df_iris,
    model="svm",
    problem_type="classification",
    model_params=model_params,
    n_jobs=2,
    verbose=3,
)

###############################################################################
# We can also parallelize over the hyperparameter search/inner cv.
# This will work by using the ``n_jobs`` argument of the searcher itself, e.g.,
# by default :class:`sklearn.model_selection.GridSearchCV`.
# To adjust the parameters of the search we have to use the ``search_params``
# inside of the ``model_params`` like this:
model_params = dict(
    svm__C=[1, 2, 3],
)
search_params = dict(n_jobs=2, verbose=3)

scores = run_cross_validation(
    X=X,
    y=y,
    data=df_iris,
    model="svm",
    problem_type="classification",
    model_params=model_params,
    search_params=search_params,
)

###############################################################################
# Depending on your resources you can use ``n_jobs`` for outer cv, inner cv or
# even as a ``model_parameter`` for some models like ``rf``.
# Additionally, you can also use the ``scikit-learn``'s ``parallel_backend`` for
# parallelization.
