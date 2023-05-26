.. include:: links.inc

.. _available_pipeline_steps:

####################################
Overview of available Pipeline Steps
####################################

The following is a list of all available steps that can be used to create
a pipeline by name. The overview is sorted based on the type of the step: 
:ref:`available_transformers` or :ref:`available_models`.

The column 'Name (str)' refers to the string-name of 
the respective step, i.e. how it should be specified when passed to e.g. the
``PipelineCreator``. The column 'Description' gives a short 
description of what the step is doing. The column 'Class' either indicates the 
underlying `scikit-learn`_ class of the respective pipeline-step together with 
a link to the class in the `scikit-learn`_ documentation (follow the link to 
see the valid parameters) or indicates the class in 
the Julearn code, so one can have a closer look at it in Julearn's 
:ref:`api`.

For feature transformations the :ref:`available_transformers` have to be used 
with the ``PipelineCreator`` and for target transformation with the 
``TargetPipelineCreator``.

.. _available_transformers:

Transformers
============

.. _available_scalers:

Scalers
-------

.. list-table::
   :widths: 30 80 40
   :header-rows: 1

   * - Name (str)
     - Description
     - Class
   * - ``zscore``
     - Removing mean and scale to unit variance
     - :class:`~sklearn.preprocessing.StandardScaler`
   * - ``scaler_robust``
     - Removing median and scale to IQR
     - `RobustScaler`_
   * - ``scaler_minmax``
     - Scale to a given range
     - `MinMaxScaler`_
   * - ``scaler_maxabs``
     - Scale by max absolute value
     - `MaxAbsScaler`_
   * - ``scaler_normalizer``
     - Normalize to unit norm
     - `Normalizer`_
   * - ``scaler_quantile``
     - Transform to uniform or normal distribution (robust)
     - `QuantileTransformer`_
   * - ``scaler_power``
     - *Gaussianise* data
     - `PowerTransformer`_


Feature Selection
-----------------
.. list-table::
   :widths: 30 80 40
   :header-rows: 1

   * - Name (str)
     - Description
     - Class
   * - ``select_univariate``
     - Removing mean and scale to unit variance
     - `GenericUnivariateSelect`_
   * - ``select_percentile``
     - Rank and select percentile
     - `SelectPercentile`_
   * - ``select_k``
     - Rank and select K
     - `SelectKBest`_
   * - ``select_fdr``
     - Select based on estimated FDR
     - `SelectFdr`_
   * - ``select_fpr``
     - Select based on FPR threshold
     - `SelectFpr`_
   * - ``select_fwe``
     - Select based on FWE threshold
     - `SelectFwe`_
   * - ``select_variance``
     - Remove low variance features
     - `VarianceThreshold`_


DataFrame operations
--------------------
.. list-table::
   :widths: 30 80 40
   :header-rows: 1

   * - Name (str)
     - Description
     - Class
   * - ``remove_confound``
     - Removing confounds from features,
       by subtracting the prediction of each feature given all confounds.
       By default this is equal to "independently regressing out 
       the confounds from the features" 
     - confounds.ConfoundRemover
   * - ``drop_columns``
     - Drop columns from the dataframe
     - dataframe.DropColumns
   * - ``change_column_types``
     - Change the type of a column in a dataframe
     - dataframe.ChangeColumnTypes
   * - ``filter_columns``
     - Filter columns in a dataframe
     - dataframe.FilterColumns

.. _available_decompositions:

Decomposition
-------------
.. list-table::
   :widths: 30 80 40
   :header-rows: 1

   * - Name (str)
     - Description
     - Class
   * - ``pca``
     - Principal Component Analysis
     - `PCA`
  
Custom
------
.. list-table::
   :widths: 30 80 40
   :header-rows: 1

   * - Name (str)
     - Description
     - Class
   * - ``cbpm``
     - Connectome-based Predictive Modeling (CBPM) 
     - cbpm.CBPM

.. _available_models:

Models (Estimators)
===================


Support Vector Machines
-----------------------
.. list-table::
   :widths: 30 80 40 20 20 20
   :header-rows: 1

   * - Name (str)
     - Description
     - Class
     - Binary
     - Multiclass
     - Regression
   * - ``svm``
     - Support Vector Machine
     - `SVC`_ and `SVR`_
     - Y
     - Y
     - Y


Ensemble
--------
.. list-table::
   :widths: 30 30 70 20 20 20
   :header-rows: 1

   * - Name (str)
     - Description
     - Class
     - Binary
     - Multiclass
     - Regression
   * - ``rf``
     - Random Forest
     - `RandomForestClassifier`_ and `RandomForestRegressor`_
     - Y
     - Y
     - Y
   * - ``et``
     - Extra-Trees
     - `ExtraTreesClassifier`_ and `ExtraTreesRegressor`_
     - Y
     - Y
     - Y
   * - ``adaboost``
     - AdaBoost
     - `AdaBoostClassifier`_ and `AdaBoostRegressor`_
     - Y
     - Y
     - Y
   * - ``bagging``
     - Bagging
     - `BaggingClassifier`_ and `BaggingRegressor`_
     - Y
     - Y
     - Y
   * - ``gradientboost``
     - Gradient Boosting 
     - `GradientBoostingClassifier`_ and `GradientBoostingRegressor`_
     - Y
     - Y
     - Y
   * - ``stacking``
     - Stacking
     - `StackingClassifier`_ and `StackingRegressor`_
     - Y
     - Y
     - Y


Gaussian Processes
------------------
.. list-table::
   :widths: 30 30 70 20 20 20
   :header-rows: 1

   * - Name (str)
     - Description
     - Class
     - Binary
     - Multiclass
     - Regression
   * - ``gauss``
     - Gaussian Process
     - `GaussianProcessClassifier`_ and `GaussianProcessRegressor`_
     - Y
     - Y
     - Y

Linear Models
-------------
.. list-table::
   :widths: 30 50 70 10 10 10
   :header-rows: 1

   * - Name (str)
     - Description
     - Class
     - Binary
     - Multiclass
     - Regression
   * - ``logit``
     - Logistic Regression (aka logit, MaxEnt).
     - `LogisticRegression`_
     - Y
     - Y
     - N
   * - ``logitcv``
     - Logistic Regression CV (aka logit, MaxEnt).
     - `LogisticRegressionCV`_
     - Y
     - Y
     - N
   * - ``linreg``
     - Least Squares regression.
     - `LinearRegression`_
     - N
     - N
     - Y
   * - ``ridge``
     - Linear least squares with l2 regularization.
     - `RidgeClassifier`_ and `Ridge`_
     - Y
     - Y
     - Y
   * - ``ridgecv``
     - Ridge regression with built-in cross-validation.
     - `RidgeClassifierCV`_ and `RidgeCV`_
     - Y
     - Y
     - Y
   * - ``sgd``
     - Linear model fitted by minimizing a regularized empirical loss with SGD
     - `SGDClassifier`_ and `SGDRegressor`_
     - Y
     - Y
     - Y


Naive Bayes
-----------
.. list-table::
   :widths: 30 50 70 10 10 10
   :header-rows: 1

   * - Name (str)
     - Description
     - Class
     - Binary
     - Multiclass
     - Regression
   * - ``nb_bernoulli``
     - Multivariate Bernoulli models.
     - `BernoulliNB`_
     - Y
     - Y
     - N
   * - ``nb_categorical``
     - Categorical features.
     - `CategoricalNB`_
     - Y
     - Y
     - N
   * - ``nb_complement``
     - Complement Naive Bayes
     - `ComplementNB`_
     - Y
     - Y
     - N
   * - ``nb_gaussian``
     - Gaussian Naive Bayes 
     - `GaussianNB`_
     - Y
     - Y
     - N
   * - ``nb_multinomial``
     - Multinomial models
     - `MultinomialNB`_
     - Y
     - Y
     - N


Dynamic Selection
-----------------
.. list-table::
   :widths: 30 50 70 10 10 10
   :header-rows: 1

   * - Name (str)
     - Description
     - Class
     - Binary
     - Multiclass
     - Regression
   * - ``ds``
     - Support for `DESlib`_ models
     - dynamic.DynamicSelection
     - Y
     - Y
     - Y


Dummy
-----
.. list-table::
   :widths: 30 50 70 10 10 10
   :header-rows: 1

   * - Name (str)
     - Description
     - Class
     - Binary
     - Multiclass
     - Regression
   * - ``dummy``
     - Use simple rules (without features).
     - `DummyClassifier`_ and `DummyRegressor`_
     - Y
     - Y
     - Y
