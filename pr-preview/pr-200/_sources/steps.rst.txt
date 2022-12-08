.. include:: links.inc

Available Pipeline Steps
========================

The following is a list of all the available steps that can be used to create
a pipeline by name.

Features Preprocessing
----------------------

Scalers
^^^^^^^

.. list-table::
   :widths: 30 80 40
   :header-rows: 1

   * - Name (str)
     - Description
     - Class
   * - ``zscore``
     - Removing mean and scale to unit variance
     - `StandardScaler`_
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
^^^^^^^^^^^^^^^^^
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


Confound Removal
^^^^^^^^^^^^^^^^
.. list-table::
   :widths: 30 80 40
   :header-rows: 1

   * - Name (str)
     - Description
     - Class
   * - ``remove_confound``
     - removing confounds from features,
       by subtracting the prediction of each feature given all confounds.
       By default this is equal to "independently regressing out 
       the confounds from the features" 
     - :class:`.confounds.DataFrameConfoundRemover`

Decomposition
^^^^^^^^^^^^^
.. list-table::
   :widths: 30 80 40
   :header-rows: 1

   * - Name (str)
     - Description
     - Class
   * - ``pca``
     - Principal Component Analysis
     - `PCA`_

Target Preprocessing
--------------------

Target Scalers
^^^^^^^^^^^^^^

.. list-table::
   :widths: 30 80 40
   :header-rows: 1

   * - Name (str)
     - Description
     - Class
   * - ``zscore``
     - Removing mean and scale to unit variance
     - `StandardScaler`_

Target Confound Removal
^^^^^^^^^^^^^^^^^^^^^^^
.. list-table::
   :widths: 30 80 40
   :header-rows: 1

   * - Name (str)
     - Description
     - Class
   * - ``remove_confound``
     - removing confounds from target,
       by subtracting the prediction of the target given all confounds.
       By default this is equal to "regressing out 
       the confounds from the target"
     - :class:`.TargetConfoundRemover`

Models
------


Support Vector Machines
^^^^^^^^^^^^^^^^^^^^^^^
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
^^^^^^^^
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


Gaussian Processes
^^^^^^^^^^^^^^^^^^
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
^^^^^^^^^^^^^
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
^^^^^^^^^^^
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

Dummy
^^^^^
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