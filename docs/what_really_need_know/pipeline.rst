.. .. include:: ../links.inc

.. .. _pipeline_usage:

.. Pipeline
.. ========

.. Content to treat: model preprocessing, model types, hyperparameter (without tuning),
.. link to examples?/one wuick pipeline?, PipelineCreator

.. .. include:: links.inc

.. Understanding the pipeline
.. ==========================
.. Julearn aims to provide an user-friendly way to apply complex machine learning
.. pipelines.
.. To do so julearn provides the :func:`.run_cross_validation` function.
.. Here, users can specify and customize their pipeline,
.. how should be fitted and evaluated.
.. Furthermore, this function allows you to return the
.. complete and fitted pipeline to use it on other data.
.. In this part of the documentation we will have a closer look to these
.. features of julearn and how you can use them.

.. .. note::
..   You should have read the :doc:`input <input>` section before.

.. Model & Problem Type
.. ********************

.. When using :func:`.run_cross_validation` you have to answer at least
..  2 questions first and then specify the according arguments properly:

..   * What model do you want to use?
..     You can enter any model name from :doc:`steps <steps>` 
..     or use any scikit-learning compatible model into the `model`
..     argument from :func:`.run_cross_validation` 
..   * What problem type do you want to answer?
..     In machine learning their are different problems you want to handle.
..     Julearn supports ``classification`` and ``regression`` problems.
..     You shout set ``problem_type`` to one of these
..     2 problem types. By default, julearn uses the ``classification``
..     type.

.. What model do you want to use and what problem type do you want to use
.. machine learning on.


.. Preprocessing
.. *************

.. Concepts
.. ^^^^^^^^

.. By default users do not have to specify how to preprocess their data.
.. In this case, julearn automatically standardizes the continuous features,
.. the confounds and removes existing confound from the continuous features.

.. But users can configure :func:`.run_cross_validation` by specifying
.. the 3 preprocessing arguments for transforming the
.. confounds, target and features respectively (in this order).

.. To do so you can set the following arguments in the
.. :func:`.run_cross_validation` :

..   * ``preprocess_X``: specifies how to transform the features.
..     Here, you can enter the names or a list of the names of available
..     transformers (:doc:`steps <steps>` ).
..     These are then applied in order to the features.
..     By default most transformers are applied only to the continuous features.
..     For more information on this and how to modify this behavior see below.

..     E.g. ``['zscore', 'pca']`` would mean that the (continuous) features are
..     first z-standardized and then reduced using a principle component analysis.
..     By default features will will not be preprocessed and confound removed in 
..     case a confound was specified.

..   * ``preprocess_confounds``: specifies how to transform the confounds.
..     Here, you use the same lists of available transformers as in 
..     ``preprocess_X``. By default confounds will not be preprocessed.

.. Example
.. ^^^^^^^
.. Assume we want not preprocess the confounds, zscore the transformer
.. and then use a pca on the features before removing the confound from these
.. features. All of these operations are included in the :doc:`steps <steps>` 
.. and can therefore be referred to by name.

.. In other words we need to set:
..   * ``preprocess_target = 'zscore'``
..   * ``preprocess_X = ['pca', 'remove_confound']``

.. Additionally, we know that we are facing a multiclass classification problem
.. and want to use a svm model.
.. Put together with an example from the :doc:`input <input>` the code looks
.. like this:

.. .. code-block:: python

..     from seaborn import load_dataset
..     from julearn import run_cross_validation

..     df_iris = load_dataset('iris')
..     X = ['sepal_length', 'sepal_width', 'petal_length']
..     y = 'species'
..     confounds = 'petal_width'

..     preprocess_confounds = []
..     preprocess_target = 'zscore'
..     preprocess_X = ['pca', 'remove_confound']
..     run_cross_validation(
..       X=X, y=y, data=df_iris, confounds=confounds,
..       model='svm', problem_type='classification',
..       preprocess_X=preprocess_X,
..       preprocess_confounds=preprocess_confounds,
..       preprocess_target=preprocess_target)



.. .. note::
..   Instead of using the name of the available transformers you can also use
..   scikit-learn compatible transformers.
..   But it is recommended to register your own transformers first.
..   For more information see (#TODO)



.. More information
.. ^^^^^^^^^^^^^^^^

.. As mentioned above julearn allows the user to specify to which variable/columns
.. or variable/column types each transformer will be applied. To do so you
.. can adjust the ``apply_to`` hyperparameter which is added to all transformers
.. used in ``preprocess_X``. You can find such an example at #TODO
.. and find more information on hyperparameter tuning in
.. :doc:`hyperparameters <hyperparameters>` .


.. The returned pipeline
.. *********************

.. The :func:`.run_cross_validation` uses all the information mentioned above
.. to create one ExtendedDataFramePipeline which is then used for
.. cross_validation. Additionally, it can return the fitted pipeline for other
.. application. E.g. you could want to test the pipeline on one additional
.. test set. But how can you do that?

.. Returning the (extended) pipeline
.. ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. There are multiple options which you can use to return the pipeline(s).
.. For all of them you have to set the `return_estimator`.
.. These are the possible options:

..   * None: Does not return any estimator
..   * ``'final'``: Return the estimator fitted on all the data.
..   * ``'cv'``: Return the all the estimator from each CV split, fitted on the
..     training data.
..   * ``'all'``: Return all the estimators (final and cv).

.. These returned estimators are always ExtendedDataFramePipeline 
.. objects.Therefore, the next section will discuss how you can use
.. a returned estimator.

.. ExtendedDataFramePipeline
.. ^^^^^^^^^^^^^^^^^^^^^^^^^
.. The ExtendedDataFramePipeline has the same basic functionality as
.. all scikit-learn pipelines or estimators, but also has some caveats.

.. Where ExtendedDataFramePipeline behave as usual
.. -----------------------------------------------

.. The following methods work as in sklearn:

..   * ``.fit()``
..   * ``.predict()``
..   * ``.score()``
..   * ``.predict_proba()``

.. Caveats ExtendedDataFramePipeline
.. ---------------------------------

.. In contrast to scikit-learn pipelines ExtendedDataFramePipeline
.. can change the ground truth (transform the target).
.. This means that any any function which uses sklearn scorer functions instead of
.. calling ``.score()`` on the ExtendedDataFramePipeline can give you
.. the wrong output without **any warning**.
.. For example `cross_validate` function of sklearn when using another scorer.

.. If you want to use such functions, you can follow this example (#TODO) which
.. shows how to use julearns ``extended_scorer`` instead


.. Additional functionality
.. ------------------------
.. Furthermore, ExtendedDataFramePipeline  have the following
.. added methods:

..   * ``preprocess``: a method to apply preprocessing steps of the pipeline to
..     some data. Furthermore, the ``until`` argument can be used to
..     only preprocess up to a specific transformer.



.. .. include:: links.inc

.. Available Pipeline Steps
.. ========================

.. The following is a list of all the available steps that can be used to create
.. a pipeline by name.

.. Features Preprocessing
.. ----------------------

.. Scalers
.. ^^^^^^^

.. .. list-table::
..    :widths: 30 80 40
..    :header-rows: 1

..    * - Name (str)
..      - Description
..      - Class
..    * - ``zscore``
..      - Removing mean and scale to unit variance
..      - `StandardScaler`_
..    * - ``scaler_robust``
..      - Removing median and scale to IQR
..      - `RobustScaler`_
..    * - ``scaler_minmax``
..      - Scale to a given range
..      - `MinMaxScaler`_
..    * - ``scaler_maxabs``
..      - Scale by max absolute value
..      - `MaxAbsScaler`_
..    * - ``scaler_normalizer``
..      - Normalize to unit norm
..      - `Normalizer`_
..    * - ``scaler_quantile``
..      - Transform to uniform or normal distribution (robust)
..      - `QuantileTransformer`_
..    * - ``scaler_power``
..      - *Gaussianise* data
..      - `PowerTransformer`_


.. Feature Selection
.. ^^^^^^^^^^^^^^^^^
.. .. list-table::
..    :widths: 30 80 40
..    :header-rows: 1

..    * - Name (str)
..      - Description
..      - Class
..    * - ``select_univariate``
..      - Removing mean and scale to unit variance
..      - `GenericUnivariateSelect`_
..    * - ``select_percentile``
..      - Rank and select percentile
..      - `SelectPercentile`_
..    * - ``select_k``
..      - Rank and select K
..      - `SelectKBest`_
..    * - ``select_fdr``
..      - Select based on estimated FDR
..      - `SelectFdr`_
..    * - ``select_fpr``
..      - Select based on FPR threshold
..      - `SelectFpr`_
..    * - ``select_fwe``
..      - Select based on FWE threshold
..      - `SelectFwe`_
..    * - ``select_variance``
..      - Remove low variance features
..      - `VarianceThreshold`_


.. Confound Removal
.. ^^^^^^^^^^^^^^^^
.. .. list-table::
..    :widths: 30 80 40
..    :header-rows: 1

..    * - Name (str)
..      - Description
..      - Class
..    * - ``remove_confound``
..      - removing confounds from features,
..        by subtracting the prediction of each feature given all confounds.
..        By default this is equal to "independently regressing out 
..        the confounds from the features" 
..      - confounds.ConfoundRemover

.. Decomposition
.. ^^^^^^^^^^^^^
.. .. list-table::
..    :widths: 30 80 40
..    :header-rows: 1

..    * - Name (str)
..      - Description
..      - Class
..    * - ``pca``
..      - Principal Component Analysis
..      - `PCA`_

.. Target Preprocessing
.. --------------------

.. Target Scalers
.. ^^^^^^^^^^^^^^

.. .. list-table::
..    :widths: 30 80 40
..    :header-rows: 1

..    * - Name (str)
..      - Description
..      - Class
..    * - ``zscore``
..      - Removing mean and scale to unit variance
..      - `StandardScaler`_

.. Target Confound Removal
.. ^^^^^^^^^^^^^^^^^^^^^^^
.. .. list-table::
..    :widths: 30 80 40
..    :header-rows: 1

..    * - Name (str)
..      - Description
..      - Class
..    * - ``remove_confound``
..      - removing confounds from target,
..        by subtracting the prediction of the target given all confounds.
..        By default this is equal to "regressing out 
..        the confounds from the target"
..      - TargetConfoundRemover

.. Models
.. ------


.. Support Vector Machines
.. ^^^^^^^^^^^^^^^^^^^^^^^
.. .. list-table::
..    :widths: 30 80 40 20 20 20
..    :header-rows: 1

..    * - Name (str)
..      - Description
..      - Class
..      - Binary
..      - Multiclass
..      - Regression
..    * - ``svm``
..      - Support Vector Machine
..      - `SVC`_ and `SVR`_
..      - Y
..      - Y
..      - Y


.. Ensemble
.. ^^^^^^^^
.. .. list-table::
..    :widths: 30 30 70 20 20 20
..    :header-rows: 1

..    * - Name (str)
..      - Description
..      - Class
..      - Binary
..      - Multiclass
..      - Regression
..    * - ``rf``
..      - Random Forest
..      - `RandomForestClassifier`_ and `RandomForestRegressor`_
..      - Y
..      - Y
..      - Y
..    * - ``et``
..      - Extra-Trees
..      - `ExtraTreesClassifier`_ and `ExtraTreesRegressor`_
..      - Y
..      - Y
..      - Y
..    * - ``adaboost``
..      - AdaBoost
..      - `AdaBoostClassifier`_ and `AdaBoostRegressor`_
..      - Y
..      - Y
..      - Y
..    * - ``bagging``
..      - Bagging
..      - `BaggingClassifier`_ and `BaggingRegressor`_
..      - Y
..      - Y
..      - Y
..    * - ``gradientboost``
..      - Gradient Boosting 
..      - `GradientBoostingClassifier`_ and `GradientBoostingRegressor`_
..      - Y
..      - Y
..      - Y


.. Gaussian Processes
.. ^^^^^^^^^^^^^^^^^^
.. .. list-table::
..    :widths: 30 30 70 20 20 20
..    :header-rows: 1

..    * - Name (str)
..      - Description
..      - Class
..      - Binary
..      - Multiclass
..      - Regression
..    * - ``gauss``
..      - Gaussian Process
..      - `GaussianProcessClassifier`_ and `GaussianProcessRegressor`_
..      - Y
..      - Y
..      - Y

.. Linear Models
.. ^^^^^^^^^^^^^
.. .. list-table::
..    :widths: 30 50 70 10 10 10
..    :header-rows: 1

..    * - Name (str)
..      - Description
..      - Class
..      - Binary
..      - Multiclass
..      - Regression
..    * - ``logit``
..      - Logistic Regression (aka logit, MaxEnt).
..      - `LogisticRegression`_
..      - Y
..      - Y
..      - N
..    * - ``logitcv``
..      - Logistic Regression CV (aka logit, MaxEnt).
..      - `LogisticRegressionCV`_
..      - Y
..      - Y
..      - N
..    * - ``linreg``
..      - Least Squares regression.
..      - `LinearRegression`_
..      - N
..      - N
..      - Y
..    * - ``ridge``
..      - Linear least squares with l2 regularization.
..      - `RidgeClassifier`_ and `Ridge`_
..      - Y
..      - Y
..      - Y
..    * - ``ridgecv``
..      - Ridge regression with built-in cross-validation.
..      - `RidgeClassifierCV`_ and `RidgeCV`_
..      - Y
..      - Y
..      - Y
..    * - ``sgd``
..      - Linear model fitted by minimizing a regularized empirical loss with SGD
..      - `SGDClassifier`_ and `SGDRegressor`_
..      - Y
..      - Y
..      - Y


.. Naive Bayes
.. ^^^^^^^^^^^
.. .. list-table::
..    :widths: 30 50 70 10 10 10
..    :header-rows: 1

..    * - Name (str)
..      - Description
..      - Class
..      - Binary
..      - Multiclass
..      - Regression
..    * - ``nb_bernoulli``
..      - Multivariate Bernoulli models.
..      - `BernoulliNB`_
..      - Y
..      - Y
..      - N
..    * - ``nb_categorical``
..      - Categorical features.
..      - `CategoricalNB`_
..      - Y
..      - Y
..      - N
..    * - ``nb_complement``
..      - Complement Naive Bayes
..      - `ComplementNB`_
..      - Y
..      - Y
..      - N
..    * - ``nb_gaussian``
..      - Gaussian Naive Bayes 
..      - `GaussianNB`_
..      - Y
..      - Y
..      - N
..    * - ``nb_multinomial``
..      - Multinomial models
..      - `MultinomialNB`_
..      - Y
..      - Y
..      - N

.. Dummy
.. ^^^^^
.. .. list-table::
..    :widths: 30 50 70 10 10 10
..    :header-rows: 1

..    * - Name (str)
..      - Description
..      - Class
..      - Binary
..      - Multiclass
..      - Regression
..    * - ``dummy``
..      - Use simple rules (without features).
..      - `DummyClassifier`_ and `DummyRegressor`_
..      - Y
..      - Y
..      - Y