# Authors: Fede Raimondo <f.raimondo@fz-juelich.de>
# License: AGPL

"""
Hyperparameter Tuning
=====================

Parameters vs Hyperparameters
-----------------------------

Parameters are the values that define the model, and are learned from the data.
For example, the weights of a linear regression model are parameters. The
parameters of a model are learned during training and are not set by the user.

Hyperparameters are the values that define the model, but are not learned from
the data. For example, the regularization parameter ``C`` of a Support Vector
Machine (:class:`~sklearn.svm.SVC`) model is a hyperparameter. The
hyperparameters of a model are set by the user before training and are not
learned during training.

Let's see an example of a :class:`~sklearn.svm.SVC` model with a regularization
parameter ``C``. We will use the ``iris`` dataset, which is a dataset of
measurements of flowers.

We start by loading the dataset and setting the features and target
variables.
"""
from seaborn import load_dataset
from pprint import pprint  # To print in a pretty way

df = load_dataset("iris")
X = df.columns[:-1].tolist()
y = "species"
X_types = {"continuous": X}

# The dataset has three kind of species. We will keep two to perform a binary
# classification.
df = df[df["species"].isin(["versicolor", "virginica"])]

###############################################################################
# We can now use the :class:`.PipelineCreator` to create a pipeline with a
# :class:`~sklearn.preprocessing.RobustScaler` and a
# :class:`~sklearn.svm.SVC`, with a regularization parameter ``C`` set to
# ``0.1``.

from julearn.pipeline import PipelineCreator

creator = PipelineCreator(problem_type="classification")
creator.add("zscore")
creator.add("svm", C=0.1)

print(creator)

###############################################################################
# Hyperparameter Tuning
# ---------------------
#
# Since it is the user who sets the hyperparameters, it is important to choose
# the right values. This is not always easy, and it is common to try different
# values and see which one works best. This process is called *hyperparameter
# tuning*.
#
# Basically, hyperparameter tuning refers to testing several hyperparameter
# values and choosing the one that works best.
#
# For example, we can try different values for the regularization parameter
# ``C`` of the :class:`~sklearn.svm.SVC` model and see which one works best.

from julearn import run_cross_validation

scores1 = run_cross_validation(
    X=X,
    y=y,
    data=df,
    X_types=X_types,
    model=creator,
)

print(f"Score with C=0.1: {scores1['test_score'].mean()}")

creator2 = PipelineCreator(problem_type="classification")
creator2.add("zscore")
creator2.add("svm", C=0.01)

scores2 = run_cross_validation(
    X=X,
    y=y,
    data=df,
    X_types=X_types,
    model=creator2,
)

print(f"Score with C=0.01: {scores2['test_score'].mean()}")

###############################################################################
# We can see that the model with ``C=0.1`` works better than the model with
# ``C=0.01``. However, to be sure that ``C=0.1`` is the best value, we should
# try more values. And since this is only one hyperparameter, it is not that
# difficult. But what if we have more hyperparameters? And what if we have
# several steps in the pipeline (e.g. feature selection, PCA, etc.)?
# This is a major problem: the more hyperparameters we have, the more
# times we use the same data for training and testing. This usually gives an
# optimistic estimation of the performance of the model.
#
# To prevent this, we can use a technique called *nested cross-validation*.
# That is, we use cross-validation to *tune the hyperparameters*, and then we
# use cross-validation again to estimate the performance of the model using
# the best hyperparameters set. It is called *nested* because we first split
# the data into training and testing sets to estimate the model performance
# (outer loop), and then we split the training set into two sets to tune the
# hyperparameters (inner loop).
#
# ``julearn`` has a simple way to do hyperparameter tuning using nested cross-
# validation. When we use a :class:`.PipelineCreator` to create a pipeline,
# we can set the hyperparameters we want to tune and the values we want to try.
#
# For example, we can try different values for the regularization parameter
# ``C`` of the :class:`~sklearn.svm.SVC` model:

creator = PipelineCreator(problem_type="classification")
creator.add("zscore")
creator.add("svm", C=[0.01, 0.1, 1, 10])

print(creator)

###############################################################################
# As we can see above, the creator now shows that the ``C`` hyperparameter
# will be tuned. We can now use this creator to run cross-validation. This will
# tune the hyperparameters and estimate the performance of the model using the
# best hyperparameters set.

scores_tuned, model_tuned = run_cross_validation(
    X=X,
    y=y,
    data=df,
    X_types=X_types,
    model=creator,
    return_estimator="all",
)

print(f"Scores with best hyperparameter: {scores_tuned['test_score'].mean()}")

###############################################################################
# We can see that the model with the best hyperparameters works better than
# the model with ``C=0.1``. But what's the best hyperparameter set? We can
# see it by printing the ``model_tuned.best_params_`` variable.

pprint(model_tuned.best_params_)

###############################################################################
# We can see that the best hyperparameter set is ``C=1``. Since this
# hyperparameter was not on the boundary of the values we tried, we can
# conclude that our search for the best ``C`` value was successful.
#
# However, by checking the :class:`~sklearn.svm.SVC` documentation, we can
# see that there are more hyperparameters that we can tune. For example, for
# the default ``rbf`` kernel, we can tune the ``gamma`` hyperparameter:

creator = PipelineCreator(problem_type="classification")
creator.add("zscore")
creator.add("svm", C=[0.01, 0.1, 1, 10], gamma=[0.01, 0.1, 1, 10])

print(creator)

scores_tuned, model_tuned = run_cross_validation(
    X=X,
    y=y,
    data=df,
    X_types=X_types,
    model=creator,
    return_estimator="all",
)

print(f"Scores with best hyperparameter: {scores_tuned['test_score'].mean()}")
pprint(model_tuned.best_params_)

###############################################################################
# We can see that the best hyperparameter set is ``C=1`` and ``gamma=0.1``.
# But since ``gamma`` was on the boundary of the values we tried, we should
# try more values to be sure that we are using the best hyperparameter set.
#
# We can even give a combination of different variable types, like the words
# ``"scale"`` and ``"auto"`` for the ``gamma`` hyperparameter:
creator = PipelineCreator(problem_type="classification")
creator.add("zscore")
creator.add(
    "svm",
    C=[0.01, 0.1, 1, 10],
    gamma=[1e-5, 1e-4, 1e-3, 1e-2, "scale", "auto"],
)

print(creator)

scores_tuned, model_tuned = run_cross_validation(
    X=X,
    y=y,
    data=df,
    X_types=X_types,
    model=creator,
    return_estimator="all",
)

print(f"Scores with best hyperparameter: {scores_tuned['test_score'].mean()}")
pprint(model_tuned.best_params_)

###############################################################################
# We can even tune hyperparameters from different steps of the pipeline. Let's
# add a :class:`~sklearn.feature_selection.SelectKBest` step to the pipeline
# and tune its ``k`` hyperparameter:

creator = PipelineCreator(problem_type="classification")
creator.add("zscore")
creator.add("select_k", k=[2, 3, 4])
creator.add(
    "svm",
    C=[0.01, 0.1, 1, 10],
    gamma=[1e-3, 1e-2, 1e-1, "scale", "auto"],
)

print(creator)

scores_tuned, model_tuned = run_cross_validation(
    X=X,
    y=y,
    data=df,
    X_types=X_types,
    model=creator,
    return_estimator="all",
)

print(f"Scores with best hyperparameter: {scores_tuned['test_score'].mean()}")
pprint(model_tuned.best_params_)

###############################################################################
# But how will ``julearn`` find the optimal hyperparameter set?
#
# Searchers
# ---------
#
# ``julearn`` uses the same concept as `scikit-learn`_ to tune hyperparameters:
# it uses a *searcher* to find the best hyperparameter set. A searcher is an
# object that receives a set of hyperparameters and their values, and then
# tries to find the best combination of values for the hyperparameters using
# cross-validation.
#
# By default, ``julearn`` uses a :class:`~sklearn.model_selection.GridSearchCV`.
# This searcher is very simple. First, it construct the "grid" of
# hyperparameters to try. As we see above, we have 3 hyperparameters to tune.
# So it constructs a 3-dimentional grid with all the possible combinations of
# the hyperparameters values. The second step is to perform cross-validation
# on each of the possible combinations of hyperparameters values.
#
# Another searcher that ``julearn`` provides is the
# :class:`~sklearn.model_selection.RandomizedSearchCV`. This searcher is
# similar to the :class:`~sklearn.model_selection.GridSearchCV`, but instead
# of trying all the possible combinations of hyperparameters values, it tries
# a random subset of them. This is useful when we have a lot of hyperparameters
# to tune, since it can be very time consuming to try all the possible, as well
# as continuous parameters that can be sampled out of a distribution. For
# more information, see the
# :class:`~sklearn.model_selection.RandomizedSearchCV` documentation.

###############################################################################
#
# Tuning more than one *grid*
# ---------------------------
#
# Following our tuning of the :class:`~sklearn.svm.SVC` hyperparameters, we
# can also see that we can tune the ``kernel`` hyperparameter. This
# hyperparameter can also be "linear". Let's see how our *grid* of
# hyperparameters would look like if we add this hyperparameter:

creator = PipelineCreator(problem_type="classification")
creator.add("zscore")
creator.add(
    "svm",
    C=[0.01, 0.1, 1, 10],
    gamma=[1e-3, 1e-2, "scale", "auto"],
    kernel=["linear", "rbf"],
)
print(creator)

###############################################################################
# We can see that the *grid* of hyperparameters is now 3-dimensional. However,
# there are some combinations that don't make much sense. For example, the
# ``gamma`` hyperparameter is only used when the ``kernel`` is ``rbf``. So
# we will be trying the ``linear`` kernel with each one of the 4 different
# ``gamma`` and 4 different ``C`` values. Those are 16 unnecessary combinations.
# We can avoid this by using multiple *grids*. One grid for the ``linear``
# kernel and one grid for the ``rbf`` kernel.
#
# ``julearn`` allows to specify multiple *grid* using two different approaches.
#
# 1. Repeating the step name with different hyperparameters:

creator = PipelineCreator(problem_type="classification")
creator.add("zscore")
creator.add(
    "svm",
    C=[0.01, 0.1, 1, 10],
    gamma=[1e-3, 1e-2, "scale", "auto"],
    kernel=["rbf"],
    name="svm",
)
creator.add(
    "svm",
    C=[0.01, 0.1, 1, 10],
    kernel=["linear"],
    name="svm",
)

print(creator)

scores1, model1 = run_cross_validation(
    X=X,
    y=y,
    data=df,
    X_types=X_types,
    model=creator,
    return_estimator="all",
)

print(f"Scores with best hyperparameter: {scores1['test_score'].mean()}")
pprint(model1.best_params_)

###############################################################################
# .. important::
#    Note that the ``name`` parameter is required when repeating a step name.
#    If we do not specify the ``name`` parameter, ``julearn`` will
#    auto-determine the step name in an unique way. The only way to force repated
#    names is to do so explicitly.

###############################################################################
# 2. Using multiple pipeline creators:

creator1 = PipelineCreator(problem_type="classification")
creator1.add("zscore")
creator1.add(
    "svm",
    C=[0.01, 0.1, 1, 10],
    gamma=[1e-3, 1e-2, "scale", "auto"],
    kernel=["rbf"],
)

creator2 = PipelineCreator(problem_type="classification")
creator2.add("zscore")
creator2.add(
    "svm",
    C=[0.01, 0.1, 1, 10],
    kernel=["linear"],
)

scores2, model2 = run_cross_validation(
    X=X,
    y=y,
    data=df,
    X_types=X_types,
    model=[creator1, creator2],
    return_estimator="all",
)


print(f"Scores with best hyperparameter: {scores2['test_score'].mean()}")
pprint(model2.best_params_)

###############################################################################
# .. important::
#    All the pipeline creators must have the same problem type and steps names
#    in order for this approach to work.

###############################################################################
# Indeed, if we compare both approaches, we can see that they are equivalent.
# They both produce the same *grid* of hyperparameters:

pprint(model1.param_grid)
pprint(model2.param_grid)

###############################################################################
# Models as hyperparameters
# -------------------------
#
# But why stop there? Models can also be considered as hyperparameters. For
# example, we can try different models for the classification task. Let's
# try the :class:`~sklearn.ensemble.RandomForestClassifier` and the
# :class:`~sklearn.linear_model.LogisticRegression` too:
#

creator1 = PipelineCreator(problem_type="classification")
creator1.add("zscore")
creator1.add(
    "svm",
    C=[0.01, 0.1, 1, 10],
    gamma=[1e-3, 1e-2, "scale", "auto"],
    kernel=["rbf"],
    name="model",
)

creator2 = PipelineCreator(problem_type="classification")
creator2.add("zscore")
creator2.add(
    "svm",
    C=[0.01, 0.1, 1, 10],
    kernel=["linear"],
    name="model",
)

creator3 = PipelineCreator(problem_type="classification")
creator3.add("zscore")
creator3.add(
    "rf",
    max_depth=[2, 3, 4],
    name="model",
)

creator4 = PipelineCreator(problem_type="classification")
creator4.add("zscore")
creator4.add(
    "logit",
    penalty=["l2", "l1"],
    dual=[False],
    solver="liblinear",
    name="model",
)

scores3, model3 = run_cross_validation(
    X=X,
    y=y,
    data=df,
    X_types=X_types,
    model=[creator1, creator2, creator3, creator4],
    return_estimator="all",
)


print(f"Scores with best hyperparameter: {scores3['test_score'].mean()}")
pprint(model3.best_params_)

###############################################################################
# Well, it seems that nothing can beat the :class:`~sklearn.svm.SVC` with
# ``kernel="rbf"`` for our classification example.
