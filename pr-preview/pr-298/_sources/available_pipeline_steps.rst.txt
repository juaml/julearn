.. include:: links.inc

.. _available_pipeline_steps:

Overview of available Pipeline Steps
====================================

The following is a list of all available steps that can be used to create
a pipeline by name. The overview is sorted based on the type of the step:
:ref:`available_transformers` or :ref:`available_models`.

* The column ``Name`` refers to the string-name of
  the respective step, i.e. how it should be specified when passed to e.g., the
  :class:`.PipelineCreator`.

* The column ``Description`` gives a short
  description of what the step is doing.

* The column ``Class`` either indicates the underlying `scikit-learn`_ class of
  the respective pipeline step together with a link to the class in the
  `scikit-learn`_ documentation (follow the link to see the valid parameters) or
  indicates the class in ``julearn``, so one can have a closer look at it in
  ``julearn``'s :ref:`api`.

For feature transformations, the :ref:`available_transformers` are to be used
with the :class:`.PipelineCreator` and for target transformations, the
:ref:`available_transformers` are to be used with the
:class:`.TargetPipelineCreator`.

.. _available_transformers:

Transformers
------------

.. _available_scalers:

Scalers
~~~~~~~

.. list-table::
   :widths: auto
   :header-rows: 1

   * - Name
     - Description
     - Class
   * - ``zscore``
     - Removing mean and scale to unit variance
     - :class:`~sklearn.preprocessing.StandardScaler`
   * - ``scaler_robust``
     - Removing median and scale to IQR
     - :class:`~sklearn.preprocessing.RobustScaler`
   * - ``scaler_minmax``
     - Scale to a given range
     - :class:`~sklearn.preprocessing.MinMaxScaler`
   * - ``scaler_maxabs``
     - Scale by max absolute value
     - :class:`~sklearn.preprocessing.MaxAbsScaler`
   * - ``scaler_normalizer``
     - Normalize to unit norm
     - :class:`~sklearn.preprocessing.Normalizer`
   * - ``scaler_quantile``
     - Transform to uniform or normal distribution (robust)
     - :class:`~sklearn.preprocessing.QuantileTransformer`
   * - ``scaler_power``
     - *Gaussianise* data
     - :class:`~sklearn.preprocessing.PowerTransformer`

Feature Selection
~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: auto
   :header-rows: 1

   * - Name
     - Description
     - Class
   * - ``select_univariate``
     - Removing mean and scale to unit variance
     - :class:`~sklearn.feature_selection.GenericUnivariateSelect`
   * - ``select_percentile``
     - Rank and select percentile
     - :class:`~sklearn.feature_selection.SelectPercentile`
   * - ``select_k``
     - Rank and select K
     - :class:`~sklearn.feature_selection.SelectKBest`
   * - ``select_fdr``
     - Select based on estimated FDR
     - :class:`~sklearn.feature_selection.SelectFdr`
   * - ``select_fpr``
     - Select based on FPR threshold
     - :class:`~sklearn.feature_selection.SelectFpr`
   * - ``select_fwe``
     - Select based on FWE threshold
     - :class:`~sklearn.feature_selection.SelectFwe`
   * - ``select_variance``
     - Remove low variance features
     - :class:`~sklearn.feature_selection.VarianceThreshold`

DataFrame operations
~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: auto
   :header-rows: 1

   * - Name
     - Description
     - Class
   * - ``confound_removal``
     - | Removing confounds from features,
       | by subtracting the prediction of each feature given all confounds.
       | By default this is equal to "independently regressing out
       | the confounds from the features"
     - :class:`.ConfoundRemover`
   * - ``drop_columns``
     - Drop columns from the DataFrame
     - :class:`.DropColumns`
   * - ``change_column_types``
     - Change the type of a column in a DataFrame
     - :class:`.ChangeColumnTypes`
   * - ``filter_columns``
     - Filter columns in a DataFrame
     - :class:`.FilterColumns`

.. _available_decompositions:

Decomposition
~~~~~~~~~~~~~

.. list-table::
   :widths: auto
   :header-rows: 1

   * - Name
     - Description
     - Class
   * - ``pca``
     - Principal Component Analysis
     - :class:`~sklearn.decomposition.PCA`

Custom
~~~~~~

.. list-table::
   :widths: auto
   :header-rows: 1

   * - Name
     - Description
     - Class
   * - ``cbpm``
     - Connectome-based Predictive Modeling (CBPM)
     - :class:`.CBPM`

.. _available_models:

Models (Estimators)
-------------------

Support Vector Machines
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: auto
   :header-rows: 1

   * - Name
     - Description
     - Class
     - Binary
     - Multiclass
     - Regression
   * - ``svm``
     - Support Vector Machine
     - | :class:`~sklearn.svm.SVC` and
       | :class:`~sklearn.svm.SVR`
     - Y
     - Y
     - Y

Ensemble
~~~~~~~~

.. list-table::
   :widths: auto
   :header-rows: 1

   * - Name
     - Description
     - Class
     - Binary
     - Multiclass
     - Regression
   * - ``rf``
     - Random Forest
     - | :class:`~sklearn.ensemble.RandomForestClassifier` and
       | :class:`~sklearn.ensemble.RandomForestRegressor`
     - Y
     - Y
     - Y
   * - ``et``
     - Extra-Trees
     - | :class:`~sklearn.ensemble.ExtraTreesClassifier` and
       | :class:`~sklearn.ensemble.ExtraTreesRegressor`
     - Y
     - Y
     - Y
   * - ``adaboost``
     - AdaBoost
     - | :class:`~sklearn.ensemble.AdaBoostClassifier` and
       | :class:`~sklearn.ensemble.AdaBoostRegressor`
     - Y
     - Y
     - Y
   * - ``bagging``
     - Bagging
     - | :class:`~sklearn.ensemble.BaggingClassifier` and
       | :class:`~sklearn.ensemble.BaggingRegressor`
     - Y
     - Y
     - Y
   * - ``gradientboost``
     - Gradient Boosting
     - | :class:`~sklearn.ensemble.GradientBoostingClassifier` and
       | :class:`~sklearn.ensemble.GradientBoostingRegressor`
     - Y
     - Y
     - Y
   * - ``stacking``
     - Stacking
     - | :class:`~sklearn.ensemble.StackingClassifier` and
       | :class:`~sklearn.ensemble.StackingRegressor`
     - Y
     - Y
     - Y

Gaussian Processes
~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: auto
   :header-rows: 1

   * - Name
     - Description
     - Class
     - Binary
     - Multiclass
     - Regression
   * - ``gauss``
     - Gaussian Process
     - | :class:`~sklearn.gaussian_process.GaussianProcessClassifier` and
       | :class:`~sklearn.gaussian_process.GaussianProcessRegressor`
     - Y
     - Y
     - Y

Linear Models
~~~~~~~~~~~~~

.. list-table::
   :widths: auto
   :header-rows: 1

   * - Name
     - Description
     - Class
     - Binary
     - Multiclass
     - Regression
   * - ``logit``
     - Logistic Regression (aka logit, MaxEnt).
     - :class:`~sklearn.linear_model.LogisticRegression`
     - Y
     - Y
     - N
   * - ``logitcv``
     - Logistic Regression CV (aka logit, MaxEnt).
     - :class:`~sklearn.linear_model.LogisticRegressionCV`
     - Y
     - Y
     - N
   * - ``linreg``
     - Least Squares regression.
     - :class:`~sklearn.linear_model.LinearRegression`
     - N
     - N
     - Y
   * - ``ridge``
     - Linear least squares with l2 regularization.
     - | :class:`~sklearn.linear_model.RidgeClassifier` and
       | :class:`~sklearn.linear_model.Ridge`
     - Y
     - Y
     - Y
   * - ``ridgecv``
     - Ridge regression with built-in cross-validation.
     - | :class:`~sklearn.linear_model.RidgeClassifierCV` and
       | :class:`~sklearn.linear_model.RidgeCV`
     - Y
     - Y
     - Y
   * - ``sgd``
     - Linear model fitted by minimizing a regularized empirical loss with SGD
     - | :class:`~sklearn.linear_model.SGDClassifier` and
       | :class:`~sklearn.linear_model.SGDRegressor`
     - Y
     - Y
     - Y

Naive Bayes
~~~~~~~~~~~

.. list-table::
   :widths: auto
   :header-rows: 1

   * - Name
     - Description
     - Class
     - Binary
     - Multiclass
     - Regression
   * - ``nb_bernoulli``
     - Multivariate Bernoulli models.
     - :class:`~sklearn.naive_bayes.BernoulliNB`
     - Y
     - Y
     - N
   * - ``nb_categorical``
     - Categorical features.
     - :class:`~sklearn.naive_bayes.CategoricalNB`
     - Y
     - Y
     - N
   * - ``nb_complement``
     - Complement Naive Bayes
     - :class:`~sklearn.naive_bayes.ComplementNB`
     - Y
     - Y
     - N
   * - ``nb_gaussian``
     - Gaussian Naive Bayes
     - :class:`~sklearn.naive_bayes.GaussianNB`
     - Y
     - Y
     - N
   * - ``nb_multinomial``
     - Multinomial models
     - :class:`~sklearn.naive_bayes.MultinomialNB`
     - Y
     - Y
     - N

Dynamic Selection
~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: auto
   :header-rows: 1

   * - Name
     - Description
     - Class
     - Binary
     - Multiclass
     - Regression
   * - ``ds``
     - Support for `DESlib`_ models
     - :class:`.DynamicSelection`
     - Y
     - Y
     - Y

Dummy
~~~~~

.. list-table::
   :widths: auto
   :header-rows: 1

   * - Name
     - Description
     - Class
     - Binary
     - Multiclass
     - Regression
   * - ``dummy``
     - Use simple rules (without features).
     - | :class:`~sklearn.dummy.DummyClassifier` and
       | :class:`~sklearn.dummy.DummyRegressor`
     - Y
     - Y
     - Y
