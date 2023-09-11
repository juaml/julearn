# Authors: Vera Komeyer <v.komeyer@fz-juelich.de>
#          Fede Raimondo <f.raimondo@fz-juelich.de>
# License: AGPL

"""
Model Evaluation
================

The output of :func:`.run_cross_validation`
-------------------------------------------

So far, we saw how to run a cross-validation using the :class:`.PipelineCreator`
and :func:`.run_cross_validation`. But what do we get as output from such a
pipeline?

Cross-validation scores
~~~~~~~~~~~~~~~~~~~~~~~

We consider the ``iris`` data example and one of the pipelines from the previous
section (feature z-scoring and a ``svm``).
"""
from julearn import run_cross_validation
from julearn.pipeline import PipelineCreator

from seaborn import load_dataset

# sphinx_gallery_start_ignore
from sklearn import set_config

set_config(display="diagram")
# sphinx_gallery_end_ignore

df = load_dataset("iris")
X = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
y = "species"
X_types = {
    "continuous": [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
    ]
}

# Create a pipeline
creator = PipelineCreator(problem_type="classification")
creator.add("zscore")
creator.add("svm")

# Run cross-validation
scores = run_cross_validation(
    X=X,
    y=y,
    X_types=X_types,
    data=df,
    model=creator,
)

###############################################################################
# The ``scores`` variable is a ``pandas.DataFrame`` object which contains the
# cross-validated metrics for each fold as columns and rows respectively.

print(scores)

###############################################################################
# We see that for example the ``test_score`` for the third fold is 0.933. This
# means that the model achieved a score of 0.933 on the validation set
# of this fold.
#
# We can also see more information, such as the number of samples used for
# training and testing.
#
# Cross-validation is particularly useful to inspect if a model is overfitting.
# For this purpose it is useful to not only see the test scores for each fold
# but also the training scores. This can be achieved by setting the
# ``return_train_score`` parameter to ``True`` in
# :func:`.run_cross_validation`:

scores = run_cross_validation(
    X=X,
    y=y,
    X_types=X_types,
    data=df,
    model=creator,
    return_train_score=True,
)

print(scores)

###############################################################################
# The additional column ``train_score`` indicates the score on the training
# set.
#
# For a model that is not overfitting, the training and test scores should be
# similar. In our example, the training and test scores are indeed similar.
#
# The column ``cv_mdsum`` on the first glance might appear a bit cryptic.
# This column is used in internal checks, to verify that the same CV was used
# when results are compared using ``julearn``'s provided statistical tests.
# This is nothing you need to worry about at this point.
#
# Returning a model (estimator)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Now that we saw that our model doesn't seem to overfit, we might be
# interested in checking how our model parameters look like. By setting the
# parameter ``return_estimator``, we can tell :func:`.run_cross_validation` to
# give us the models. It can have three different values:
#
# 1. ``"cv"``: This option indicates that we want to get the model that was
#    trained on the entire training data of each CV fold. This means that we
#    get as many models as we have CV folds. They will be returned within the
#    scores DataFrame.
#
# 2. ``"final"``: With this setting, an additional model will be trained on the
#    entire input dataset. This model will be returned as a separate variable.
#
# 3. ``"all"``: In this scenario, all the estimators (``"final"`` and ``"cv"``)
#    will be returned.
#
# For demonstration purposes we will have a closer look at the ``"final"``
# estimator option.

scores, model = run_cross_validation(
    X=X,
    y=y,
    X_types=X_types,
    data=df,
    model=creator,
    return_train_score=True,
    return_estimator="final",
)

print(scores)

###############################################################################
# As we see, the scores DataFrame is the same as before. However, we now have
# an additional variable ``model``. This variable contains the final estimator
# that was trained on the entire training dataset.

model

###############################################################################
# We can use this estimator object to for example inspect the coefficients of
# the model or make predictions on a hold out test set. To learn more about how
# to inspect models please have a look at :ref:`model_inspection`.
#
# Cross-validation splitters
# --------------------------
#
# When performing a cross-validation, we need to split the data into training
# and validation sets. This is done by a *cross-validation splitter*, that
# defines how the data should be split, how many folds should be used and
# whether to repeat the process several times. For example, we might want to
# shuffle the data before splitting, stratify the splits so the distribution of
# targets are always represented in the individual folds, or consider certain
# grouping variables in the splitting process, so that samples from the same
# group are always in the same fold and not split across folds.
#
# So far, however, we didn't specify anything in that regard and still the
# cross-validation was performed and we got five folds (see the five rows above
# in the scores dataframe). This is because the default behaviour in
# :func:`.run_cross_validation` falls back to the ``scikit-learn`` defaults,
# which is a :class:`sklearn.model_selection.StratifiedKFold` (with ``k=5``)
# for classification and :class:`sklearn.model_selection.KFold` (with ``k=5``)
# for regression.
#
# .. note::
#   These defaults will change when they are changed in ``scikit-learn`` as here
#   ``julearn`` uses ``scikit-learn``'s defaults.
#
# We can define the cross-validation splitting strategy ourselves by passing an
# ``int, str or cross-validation generator`` to the ``cv`` parameter of
# :func:`.run_cross_validation`. The default described above is ``cv=None``.
# the second option is to pass only an integer to ``cv``. In that case, the
# same default splitting strategies will be used
# (:class:`sklearn.model_selection.StratifiedKFold` for classification,
# :class:`sklearn.model_selection.KFold` for regression), but the number of
# folds will be changed to the value of the provided integer (e.g., ``cv=10``).
# To define the entire splitting strategy, one can pass all ``scikit-learn``
# compatible splitters :mod:`sklearn.model_selection` to ``cv``. However,
# ``julearn`` provides a built-in set of additional splitters that can be found
# under :mod:`.model_selection` (see more about them in :ref:`cv_splitter`).
# The fourth option is to pass an iterable that yields the train and test
# indices for each split.
#
# Using the same pipeline creator as above, we can define a cv-splitter and
# pass it to :func:`.run_cross_validation` as follows:

from sklearn.model_selection import RepeatedStratifiedKFold

cv_splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)

scores = run_cross_validation(
    X=X,
    y=y,
    X_types=X_types,
    data=df,
    model=creator,
    return_train_score=True,
    cv=cv_splitter,
)

###############################################################################
# This will repeat 2 times a 5-fold stratified cross-validation. So the
# returned ``scores`` DataFrame will have 10 rows. We set the ``random_state``
# to an arbitrary integer to make the splitting of the data reproducible.

print(scores)

###############################################################################
# Scoring metrics
# ---------------
#
# Nice, we have a basic pipeline, with preprocessing of features and a model,
# we defined the splitting strategy for the cross-validation the way we want it
# and we had a look at our resulting train and test scores when performing
# the cross-validation. But what do these scores even mean?
#
# Same as for the kind of cv-splitter, :func:`.run_cross_validation` has a
# default assumption for the scorer to be used to evaluate the
# cross-validation, which is always the model's default scorer. Remember, we
# used a support vector classifier with the ``y`` (target) variable being the
# species of the ``iris`` dataset (possible values: ``'setosa'``,
# ``'versicolor'`` or ``'virginica'``). Therefore we have a multi-class
# classification (not to be confused with a multi-label classification!).
# Checking ``scikit-learn``'s documentation of a support vector classifier's
# default scorer :meth:`sklearn.svm.SVC.score`, we can see that this is the
# 'mean accuracy on the given test data and labels'.
#
# With the ``scoring`` parameter of :func:`.run_cross_validation`, one can
# define the scoring function to be used. On top of the available
# ``scikit-learn`` :mod:`sklearn.metrics`, ``julearn`` extends the functionality
# with more internal scorers and the possibility to define custom scorers. To see
# the available ``julearn`` scorers, one can use the :func:`.list_scorers`
# function:

from julearn import scoring
from pprint import pprint  # for nice printing

pprint(scoring.list_scorers())

###############################################################################
# To use a ``julearn`` scorer, one can pass the name of the scorer as a string
# to the ``scoring`` parameter of :func:`.run_cross_validation`. If multiple
# different scorers need to be used, a list of strings can be passed. For
# example, if we were interested in the ``accuracy`` and the ``f1`` scores we
# could do the following:

scoring = ["accuracy", "f1_macro"]

scores = run_cross_validation(
    X=X,
    y=y,
    X_types=X_types,
    data=df,
    model=creator,
    return_train_score=True,
    cv=cv_splitter,
    scoring=scoring,
)

###############################################################################
# The ``scores`` DataFrame will now have train- and test-score columns for both
# scorers:

print(scores)

###############################################################################
# .. # TODO?
# .. Additionally, julearn allows the user to define and register any function and
# .. use it as a scorer in the same way scikit-learn or julearn internal scorers work.
# ..  <add code example?>
