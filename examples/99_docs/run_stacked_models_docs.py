# Authors: Leonard Sasse <l.sasse@fz-juelich.de>
# License: AGPL

"""
Stacking Models
===============

``scikit-learn`` already provides a stacking implementation for
:class:`stacking regression<sklearn.ensemble.StackingRegressor>` as well
as for :class:`stacking classification<sklearn.ensemble.StackingClassifier>`.

Now, ``scikit-learn``'s stacking implementation will fit each estimator on all
of the data. However, this may not always be what you want. Sometimes you want
one estimator in the ensemble to be fitted on one type of features, while fitting
another estimator on another type of features. ``julearn``'s API provides some
extra flexibility to build more flexible and customizable stacking pipelines.
In order to explore its capabilities, let's first look at this simple example
of fitting each estimator on all of the data. For example, we can stack a
support vector regression (SVR) and a random forest regression (RF) to predict
some target in a bit of toy data.

Fitting each estimator on all of the features
---------------------------------------------
First, of course, let's import some necessary packages. Let's also configure
``julearn``'s logger to get some additional information about what is happening:
"""

from sklearn.datasets import make_regression
import pandas as pd
import numpy as np

from julearn import run_cross_validation
from julearn.pipeline import PipelineCreator
from julearn.utils import configure_logging

configure_logging(level="INFO")

###############################################################################
# Now, that we have these out of the way, we can create some artificial toy
# data to demonstrate a very simple stacking estimator within ``julearn``. We
# will use a dataset with 20 features and 200 samples.

# Prepare data
X, y = make_regression(n_features=20, n_samples=200)

# Make dataframe
X_names = [f"feature_{x}" for x in range(1, 21)]
data = pd.DataFrame(X)
data.columns = X_names
data["target"] = y

###############################################################################
# To build a stacking pipeline, we have to initialize each estimator that we
# want to use in stacking, and then of course the stacking estimator itself.
# Let's start by initializing an SVR. For this we can use the
# :class:`.PipelineCreator`. Keep in mind that this is only an example, and
# the hyperparameter grids we use here are somewhat arbitrary:

model_1 = PipelineCreator(problem_type="regression", apply_to="*")
model_1.add("svm", kernel="linear", C=np.geomspace(1e-2, 1e2, 10))

###############################################################################
# Note, that we specify applying the model to all of the features using
# :code:`apply_to="*"`. Now, let's also create a pipeline for our random
# forest estimator:

model_2 = PipelineCreator(problem_type="regression", apply_to="*")
model_2.add(
    "rf",
    n_estimators=20,
    max_depth=[10, 50],
    min_samples_leaf=[1, 3, 4],
    min_samples_split=[2, 10],
)

###############################################################################
# We can now provide these two models to a :class:`.PipelineCreator` to
# initialize a stacking model. The interface for this is very similar to a
# :class:`sklearn.pipeline.Pipeline`:

# Create the stacking model
model = PipelineCreator(problem_type="regression")
model.add(
    "stacking",
    estimators=[[("model_1", model_1), ("model_2", model_2)]],
    apply_to="*",
)

###############################################################################
# This final stacking :class:`.PipelineCreator` can now simply be handed over
# to ``julearn``'s :func:`.run_cross_validation`:

scores, final = run_cross_validation(
    X=X_names,
    y="target",
    data=data,
    model=model,
    seed=200,
    return_estimator="final",
)

###############################################################################
# Fitting each estimator on a specific feature type
# -------------------------------------------------
#
# As you can see, fitting a standard ``scikit-learn`` stacking estimator is
# relatively simple with ``julearn``. However, sometimes it may be desirable to
# have a bit more control over which features are used to fit each estimator.
# For example, there may be two types of features. One of these feature types
# we may want to use for fitting the SVR, and one of these feature types we
# may want to use for fitting the RF. To demonstrate how this can be done in
# ``julearn``, let's now create some very similar toy data, but distinguish
# between two different types of features: ``"type1"`` and ``"type2"``.

# Prepare data
X, y = make_regression(n_features=20, n_samples=200)

# Prepare feature names and types
X_types = {
    "type1": [f"type1_{x}" for x in range(1, 11)],
    "type2": [f"type2_{x}" for x in range(1, 11)],
}

# First 10 features are "type1", second 10 features are "type2"
X_names = X_types["type1"] + X_types["type2"]

# Make dataframe, apply correct column names according to X_names
data = pd.DataFrame(X)
data.columns = X_names
data["target"] = y

###############################################################################
# Let's first configure a :class:`.PipelineCreator` to fit an SVR on the
# features of ``"type1"```:

model_1 = PipelineCreator(problem_type="regression", apply_to="type1")
model_1.add("filter_columns", apply_to="*", keep="type1")
model_1.add("svm", kernel="linear", C=np.geomspace(1e-2, 1e2, 10))

###############################################################################
# Afterwards, let's configure a :class:`.PipelineCreator` to fit a RF on the
# features of ``"type2"```:

model_2 = PipelineCreator(problem_type="regression", apply_to="type2")
model_2.add("filter_columns", apply_to="*", keep="type2")
model_2.add(
    "rf",
    n_estimators=20,
    max_depth=[10, 50],
    min_samples_leaf=[1, 3, 4],
    min_samples_split=[2, 10],
)

###############################################################################
# Now, as in the previous example, we only have to create a stacking estimator
# that uses both of these estimators internally. Then we can simply use this
# stacking estimator in a :func:`.run_cross_validation` call:

# Create the stacking model
model = PipelineCreator(problem_type="regression")
model.add(
    "stacking",
    estimators=[[("model_1", model_1), ("model_2", model_2)]],
    apply_to="*",
)

# Run
scores, final = run_cross_validation(
    X=X_names,
    X_types=X_types,
    y="target",
    data=data,
    model=model,
    seed=200,
    return_estimator="final",
)

###############################################################################
# As you can see, the :class:`.PipelineCreator` and the in-built
# :class:`~sklearn.ensemble.StackingRegressor` make it very easy to flexibly
# build some very powerful stacking pipelines. Of course, you can do the same
# for classification which will use the
# :class:`~sklearn.ensemble.StackingClassifier` instead.
