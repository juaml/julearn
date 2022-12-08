.. include:: links.inc

Input Data
==========

julearn supports two kinds of data input configuration. The function 
:func:`.run_cross_validation` takes as input the following variables:

- `X`: Features
- `y`: Target or labels
- `confounds`: Confounds to remove (optional)
- `pos_labels`: Labels to be considered as positive (optional, needed for some
   metrics)
- `groups`: Grouping variables to avoid data leakage in some cross-validation
   schemes. See `Cross Validation`_ for more information.

julearn interprets data using two kinds of combinations:

Using Pandas dataframes (recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This method interprets `X`, `y`, `confounds` and `groups` as columns in the
dataframe (specified in `df`).

For example, using the 'iris' dataset, we can specify:

.. code-block:: python

    df_iris = load_dataset('iris')
    X = ['sepal_length', 'sepal_width', 'petal_length']
    y = 'species'
    confounds = 'petal_width'

And finally call :func:`.run_cross_validation` with the following parameters:

.. code-block:: python

    scores = run_cross_validation(X=X, y=y, data=df_iris, confounds=confounds)

Using regular expressions
-------------------------
It might be the case that the number of elements of X and confounds are too
many to specify manually all the column names. For this purpose, julearn
provides the option of using regular expressions to match columns names.

In the previous example, we can pick both ``sepal_width`` and ``sepal_length``
by using ``sepal_.*``.

.. code-block:: python

    df_iris = load_dataset('iris')
    X = ['sepal_.*', 'petal_length']
    y = 'species'
    confounds = 'petal_width'

Additionally, we also provide a way to select all the columns, except for the
ones used for ``y``, ``confounds`` and ``groups``. That is, using X = [':'].

.. code-block:: python

    df_iris = load_dataset('iris')
    X = [':']
    y = 'species'
    confounds = 'petal_width'

For more information, check python's `Regular Expressions`_. Keep in mind that 
julearn uses `fullmatch`, so it requires that the regular expression matches
the whole string and not part of it.

Using Numpy arrays
^^^^^^^^^^^^^^^^^^
This method allows `X`, `y`, `confounds` and groups to be specified as 
n-dimensional arrays. In this case, the number of samples for `X`, `y`,
`confounds` and `groups` must match:

.. code-block:: python

    X.shape[0] == y.shape[0] == confunds.shape[0] == groups.shape[0]


`X` (and confounds) can be one- or two-dimensional, with each element in the
second dimension representing a feature (or confound):

.. code-block:: python

    if X.ndim == 1:
        n_features == 1
    else:
        n_features == X.shape[1]


Additionally, `y` and `groups` must be one-dimensional:

.. code-block:: python

    y.ndim == 1
    groups.ndim == 1

The previous example can be also written as numpy arrays:

.. code-block:: python

    df_iris = load_dataset('iris')
    features = ['sepal_length', 'sepal_width', 'petal_length']
    target = 'species'
    confound_names = 'petal_width'

    X = df_iris[features].values
    y = df_iris[target].values
    confounds = df_iris[confound_names].values

And finally call :func:`.run_cross_validation` without specifying the `df`
parameter:

.. code-block:: python

    scores = run_cross_validation(X=X, y=y, confounds=confounds)

