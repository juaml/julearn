.. include:: links.inc

Input Data
==========

While in the past, julearn supported two kinds of data input configurations, 
we have ditched the support for Nunmpy arrays in favor of Pandas dataframes.

The function :func:`.run_cross_validation` takes as input the following variables:

- `X`: Features
- `y`: Target or labels
- `pos_labels`: Labels to be considered as positive (optional, needed for some
   metrics)
- `groups`: Grouping variables to avoid data leakage in some cross-validation
   schemes. See `Cross Validation`_ for more information.


The parameters `X`, `y`, `confounds` and `groups` are interpteted as columns in the
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
