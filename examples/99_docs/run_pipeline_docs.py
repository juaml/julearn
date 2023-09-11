# Authors: Vera Komeyer <v.komeyer@fz-juelich.de>
#          Fede Raimondo <f.raimondo@fz-juelich.de>
# License: AGPL

"""
Model Building
==============

So far we know how to parametrize :func:`.run_cross_validation` in terms of the
input data (see :ref:`data_usage`). In this section, we will have a look
on how we can parametrize the learning algorithm and the preprocessing steps,
also known as the *pipeline*.

A machine learning pipeline is a process to automate the workflow of
a predictive model. It can be thought of as a combination of pipes and
filters. At a pipeline's starting point, the raw data is fed into the first
filter. The output of this filter is then fed into the next filter
(through a pipe). In supervised machine learning, different filters inside the
pipeline modify the data, while the last step is a learning algorithm that
generates predictions. Before using the pipeline to predict new data, the
pipeline has to be trained (*fitted*) on data. We call this, as ``scikit-learn``
does, *fitting* the pipeline.

``julearn`` aims to provide a user-friendly way to build and evaluate complex
machine learning pipelines. The :func:`.run_cross_validation` function is the
entry point to safely evaluate pipelines by making it easy to specify,
customize and train the pipeline. We first have a look at the most
basic pipeline, only consisting of a machine learning algorithm. Then we will
make the pipeline incrementally more complex.

.. _basic_cv:

Pipeline specification in :func:`.run_cross_validation`
-------------------------------------------------------

One important aspect when building machine learning models is the selection of
a learning algorithm. This can be specified in :func:`.run_cross_validation`
by setting the ``model`` parameter. This parameter can be any ``scikit-learn``
compatible learning algorithm. However, ``julearn`` provides a list of built-in
:ref:`available_models` that can be specified by name (see ``Name`` column in
:ref:`available_models`). For example, we can simply set
``model=="svm"`` to use a Support Vector Machine (SVM) [#1]_.

Let's first specify the data parameters as we learned in :ref:`data_usage`:
"""
from julearn import run_cross_validation
from seaborn import load_dataset

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

##############################################################################
# Now we can run the cross validation with SVM as the learning algorithm:

scores = run_cross_validation(
    X=X,
    y=y,
    data=df,
    X_types=X_types,
    model="svm",
    problem_type="classification",
)
print(scores)

##############################################################################
# You will notice that this code indicates an extra parameter ``problem_type``.
# This is because in machine learning, one can distinguish between regression
# problems -when predicting a continuous outcome- and classification problems
# -for discrete class label predictions-. Therefore,
# :func:`.run_cross_validation` additionally needs to know which problem type
# we are interested in. The possible values for ``problem_type`` are
# ``classification`` and ``regression``. In the example we are interested in
# predicting the species (see ``y`` in :ref:`data_usage`), i.e. a discrete
# class label.
#
# Et voilà, your first machine learning pipeline is ready to go.

##############################################################################
# Feature preprocessing
# ---------------------
# There are cases in which the input data, and in particular the features,
# should be transformed before passing them to the learning algorithm. One
# scenario can be that certain learning algorithms need the features in a
# specific form, for example in standardized form, so that the data resemble a
# normal distribution. This can be achieved by first z-scoring (or standard
# scaling) the features (see :ref:`available_scalers`).
#
# Importantly in a machine learning workflow, all transformations done to the
# data have to be done in a cv-consistent way. That means that
# data transformation steps have to be done on the training data of each
# respective cross-validation fold and then *only* apply the parameters of the
# transformation to the validation data of the respective fold. One should
# **never** do preprocessing on the entire dataset and then do
# cross-validation on the already preprocessed features (or more
# generally transformed data) because this leads to leakage of information from
# the validation data into the model. This is exactly where
# :func:`.run_cross_validation` comes in handy, because you can simply add your
# desired preprocessing step (:ref:`available_transformers`) and it
# takes care of doing the respective transformations in a cv-consistent manner.
#
# Let's have a look at how we can add a z-scoring step to our pipeline:

scores = run_cross_validation(
    X=X,
    y=y,
    data=df,
    X_types=X_types,
    preprocess="zscore",
    model="svm",
    problem_type="classification",
)

print(scores)

##############################################################################
# .. note::
#   Learning algorithms (what we specified in the ``model`` parameter), are
#   **estimators**. Preprocessing steps however, are usually **transformers**,
#   because they transform the input data in a certain way. Therefore, the
#   parameter description in the API of :func:`.run_cross_validation`,
#   defines valid input for the ``preprocess`` parameter as
#   ``TransformerLike``::
#
#      preprocess : str, TransformerLike or list | None
#              Transformer to apply to the features. If string, use one of the
#              available transformers. If list, each element can be a string or
#              scikit-learn compatible transformer. If None (default), no
#              transformation is applied.

##############################################################################
# But what if we want to add more preprocessing steps?
# For example, in the case that there are many features available, we might
# want to reduce the dimensionality of the features before passing them to the
# learning algorithm. A commonly used approach is a principal component
# analysis (PCA, see :ref:`available_decompositions`). If we nevertheless
# want to keep our previously applied z-scoring, we can simply add the PCA as
# another preprocessing step as follows:

scores = run_cross_validation(
    X=X,
    y=y,
    data=df,
    X_types=X_types,
    preprocess=["zscore", "pca"],
    model="svm",
    problem_type="classification",
)

print(scores)

##############################################################################
# This is nice, but with more steps added to the pipeline this can become
# opaque. To simplify building complex pipelines, ``julearn`` provides a
# :class:`.PipelineCreator` which helps keeping things neat.
#
# .. _pipeline_creator:
#
# Pipeline specification made easy with the :class:`.PipelineCreator`
# -------------------------------------------------------------------
#
# The :class:`.PipelineCreator` is a class that helps the user create complex
# pipelines with straightforward usage by adding, in order, the desired steps
# to the pipeline. Once the pipeline is specified, the
# :func:`.run_cross_validation` will detect that it is a pipeline creator and
# will automatically create the pipeline and run the cross-validation.
#
# .. note::
#   The :class:`.PipelineCreator` always has to be initialized with the
#   ``problem_type`` parameter, which can be either ``classification`` or
#   ``regression``.
#
# Let's re-write the previous example, using the :class:`.PipelineCreator`.
#
# We start by creating an instance of the :class:`.PipelineCreator`, and
# setting the ``problem_type`` parameter to ``classification``.

from julearn.pipeline import PipelineCreator

creator = PipelineCreator(problem_type="classification")

##############################################################################
# Then we use the ``add`` method to add every desired step to the pipeline.
# Both, the preprocessing steps and the learning algorithm are added in the
# same way.
# As with the :func:`.run_cross_validation` function, one can use the names
# of the step as indicated in :ref:`available_pipeline_steps`.

creator.add("zscore")
creator.add("pca")
creator.add("svm")

print(creator)

##############################################################################
# We then pass the ``creator`` to :func:`.run_cross_validation` as the
# ``model`` parameter. We do not need to (and cannot) specify any other
# pipeline specification step (such as ``preprocess``)

scores = run_cross_validation(
    X=X,
    y=y,
    data=df,
    X_types=X_types,
    model=creator,
)

print(scores)

##############################################################################
# Awesome! We covered how to create a basic machine learning pipeline and
# even added multiple feature prepreprocessing steps.
#
# Let's jump to the next important aspect in the process of building a machine
# learning model: **Hyperparameters**. We here cover the basics of setting
# hyperparameters. If you want to know more about tuning (or optimizing)
# hyperparameters, please have a look at :ref:`hp_tuning`.
#
# Setting hyperparameters
# -----------------------
#
# If you are new to machine learning, the section heading might confuse you:
# Parameters, hyperparameters - aren't we doing machine learning, so shouldn't
# the model learn all our parameters? Well, yes and no. Yes, it should learn
# parameters. However, hyperparameters and parameters are two different things.
#
# A **model parameter** is a variable that is internal to the learning
# algorithm and we want to learn or estimate its value from the data, which in
# turn means that they are not set manually. They are required by the model and
# are often saved as part of the fitting process. Examples of model parameters
# are the weights in an artificial neural network, the support vectors in a
# support vector machine or the coefficients/weights in a linear or logistic
# regression.
#
# **Hyperparameters** in turn, are *configuration(s)* of a learning algorithm,
# which cannot be estimated from data, but nevertheless need to be specified to
# determine how the model parameters will be learnt. The best value for a
# hyperparameter on a given problem is usually not known and therefore has to
# be either set manually, based on experience from a previous similar problem,
# set by using a heuristic (rule of thumb) or by being *tuned*. Examples are
# the learning rate for training a neural network, the ``C`` and ``sigma``
# hyperparameters for support vector machines or the number of estimators in a
# random forest.
#
# Manually specifying hyperparameters with ``julearn`` is as simple as using the
# :class:`.PipelineCreator` and set the hyperparameter when the step is added.
#
# Let's say we want to set the ``with_mean`` parameter of the z-score
# transformer and compute PCA up to 20% of the variance explained.
# This is how the creator would look like:

creator = PipelineCreator(problem_type="classification")
creator.add("zscore", with_mean=True)
creator.add("pca", n_components=0.2)
creator.add("svm")

print(creator)

###############################################################################
# Usable transformers or estimators can be seen under
# :ref:`available_pipeline_steps`. The basis for most of these steps are the
# respective ``scikit-learn`` estimators or transformers. To see the valid
# hyperparameters for a certain transformer or estimator, just follow the
# respective link in :ref:`available_pipeline_steps` which will lead you to the
# `scikit-learn`_ documentation where you can read more about them.
#
# In many cases one wants to specify more than one hyperparameter. This can be
# done by passing each hyperparameter separated by a comma. For the ``svm`` we
# could for example specify the ``C`` and the kernel hyperparameter like this:


creator = PipelineCreator(problem_type="classification")
creator.add("zscore", with_mean=True)
creator.add("pca", n_components=0.2)
creator.add("svm", C=0.9, kernel="linear")

print(creator)

###############################################################################
# .. _apply_to_feature_types:
#
# Selective preprocessing using feature types
# -------------------------------------------
#
# Under :ref:`pipeline_creator` you might have wondered, how the
# :class:`.PipelineCreator` makes things easier. Beside the straightforward
# definition of hyperparameters, the :class:`.PipelineCreator` also allows to
# specify if a certain step must only be applied to certain features types
# (see :ref:`data_usage` on how to define feature types).
#
# In our example, we can now choose to do two PCA steps, one for the *petal*
# features, and one for the *sepal* features.
#
# First, we need to define the ``X_types`` so we have both *petal* and *sepal*
# features:

X_types = {
    "sepal": ["sepal_length", "sepal_width"],
    "petal": ["petal_length", "petal_width"],
}

###############################################################################
# Then, we modify the previous creator to add the ``pca`` step to the creator
# and specify that it should only be applied to the *petal* and *sepal*
# features. Since we also want the ``zscore`` applied to all features, we need
# to specify this as well, indicating that we want to apply it to both
# *petal* and *sepal* features:

creator = PipelineCreator(problem_type="classification")
creator.add("zscore", apply_to=["petal", "sepal"], with_mean=True)
creator.add("pca", apply_to="petal", n_components=1)
creator.add("pca", apply_to="sepal", n_components=1)
creator.add("svm")

print(creator)

###############################################################################
# We have additionally specified as a hyperparameter of the ``pca``
# that we want to use only the first component. For the ``svm`` we used
# the default hyperparameters.
#
# Finally, we again pass the defined ``X_types`` and the ``creator`` to
# :func:`.run_cross_validation`:

scores = run_cross_validation(
    X=X,
    y=y,
    data=df,
    X_types=X_types,
    model=creator,
)

print(scores)

###############################################################################
# We covered how to set up basic pipelines, how to use the
# :class:`.PipelineCreator`, how to use the ``apply_to`` parameter of the
# :class:`.PipelineCreator` and covered basics of hyperparameters. Additionally,
# we saw a basic use-case of target preprocessing. In the next
# step we will understand the returns of :func:`.run_cross_validation`, i.e.,
# the model output and the scores of the performed cross-validation.
#
# .. topic:: References:
#
#    .. [#1] Boser, B. E., Guyon, I. M., & Vapnik, V. N., `"A training
#       algorithm for optimal margin classifiers"
#       <https://dl.acm.org/doi/pdf/10.1145/130385.130401>`_, COLT '92
#       Proceedings of the fifth annual workshop on Computational learning
#       theory. 1992 Jul; 144-152.
