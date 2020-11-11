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
     - Removing mean and scale to unit variance
     - :class:`.DataFrameConfoundRemover`

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
    