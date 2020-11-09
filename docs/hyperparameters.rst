.. include:: links.inc

Hyperparameters
===============

From `Wikipedia`_:
"In machine learning, a hyperparameter is a parameter whose value is used to
control the learning process. By contrast, the values of other parameters
are derived via training."

Setting Hyperparameters
***********************

In julearn, we can set the hyperparameters using the ``model_params`` parameter
in :func:`.run_cross_validation`.

In order to specify a hyperparameter for a specific step of the pipeline, we
need to prefix the parameter name with the step name, separated by double
underscores (``__``). For example, if we want to specify the parameter
``kernel='linear'`` of the *svm* model, we need to do the following:

.. code-block:: python

    model_params = {'svm__kernel': 'linear'}
    run_cross_validation(X, y, data=df, model='svm', model_params=model_params)

TODO: Add doc for confound and y transformers hyperparameters

Tunning Hyperparameters
***********************
Another useful techinique is called *hyperparameter tunning*. This techinique
is normally applied when we want to set the hyperparameters in a data driven
way. 

In this case, we use yet another cross-validation scheme to split the training
data and evaluate the best set of hyperparameters for our model.

In julearn, hyperparameter tunning is enabled by giving more than one option
for at least one hyperparameter.

For example, obtaining the best regularization parameter (``C``) for the *svm*
model can be done by testing several candidate values:

.. code-block:: python

    model_params = {
        'svm__C': [0.001, 0.01, 0.1, 1, 10, 100],
        'svm__kernel': 'linear'}
    run_cross_validation(X, y, data=df, model='svm', model_params=model_params)

In this case, since there are more than option for C, julearn will evaluate
which is the best value. For the ``kernel`` parameter, there will be no search
as ``'linear'`` is the only value specified.

There are three more parameters that we can use to specify how the
hyperparameter search will be done:

* ``search``: The kind of algorithm to apply. The value ``'grid'`` (default)
  will apply a `GridSearchCV`. The value ``'random'`` will use a 
  `RandomizedSearchCV`_
* ``cv``: The cross-validation scheme to use for the hyperparameter tunning.
  The default is to use the same scheme as for the model evaluation.
* ``scoring``: The scoring metric to optimize.
* ``search_params``: Additional parameters for the search algorithm.

The following examples performs a randomized search with 6 options for `C` and
13 options for ``gamma``. Testing all the combinations will require 78
different evaluations. The randomised `RandomizedSearchCV`_ will not try them
all, but a number specified in ``n_iter``. In this case, we want to try 50
random combinations.

.. code-block:: python

    model_params = {
        'svm__C': [0.001, 0.01, 0.1, 1, 10, 100],
        'svm__gamma': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1,
                       10, 100, 1000],
        'svm__kernel': 'rbf',
        'search': 'random',
        'search_params': {'n_iter': 50}
        
    }
    run_cross_validation(X, y, data=df, model='svm', model_params=model_params)


For more information, see scikit-learn's `Hyperparameter Tunning`_.


