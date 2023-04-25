.. include:: ../links.inc

.. _data_usage:

Data
====

Input data to :func:`.run_cross_validation`
-------------------------------------------

Julearn deals with data in the form of pandas DataFrames. Therefore, 
:func:`.run_cross_validation` needs to know the name of the DataFrame that 
contains the features and the target or label. Additionally, one has to specify 
a list of string(s) referring to the column names of the columns containing the 
feature(s) or the target, respectively. This leads to the following 
**parameters for inputting data**:

- ``data``: Name of the dataframe containing the features and the target or label
- ``X``: List of strings containing the column names of the features
- ``y``: String containing the name of the column wiht the target or label

For example, using the well known 'iris' dataset, we can specify the data input
as follows:

First, we load the data into a pandas dataframe called ``df`` and specify 
``X`` and ``y``:

.. code-block:: python

    from seaborn import load_dataset
    
    df = load_dataset('iris')
    X = df.iloc[:,:-1].columns.tolist()
    y = "species"

Let's inspect what your variables should look like.

The dataframe:

.. code-block:: python

    df.head()

.. image:: ../images/iris_df.png
    :width: 600
    :alt: iris_df

The feature columns:

.. code-block:: python

    X

.. image:: ../images/iris_X.png
    :width: 600
    :alt: iris_X

The target column:

.. code-block:: python

    y

.. image:: ../images/iris_y.png
    :width: 600
    :alt: iris_y

Julearn's :func:`.run_cross_validation` function so far would look like this:

.. code-block:: python

    run_cross_validation(X=X, y=y, data=df)

This is not yet very useful to do machine learning, but we will come to it step 
by step.

Give the feature columns a specific type
----------------------------------------

A nice add-on that Julearn offers is to specify costum-based types for the 
features. This comes in handy, if within the machine learning workflow one 
wants to apply certain processing steps only to certain types of variables. 
We go into depth of such a scenario in :ref:`apply_to_feature_types`. If later,
for example, we want to apply certain processing steps only to the 
variables/columns related to _petal_ information, we can define ``"petal"`` in
``X_types``:

.. code-block:: python

    X_types = {"petal": ["petal_length", "petal_width"]}


The ``X_types`` can be everything, for example we could have also given the 
petal-related columns the name `some_name`:

.. code-block:: python

    X_types = {
        "some_name": ["petal_length", "petal_width"]
    }

But every column can only have **one type**!

Adding an ``X_types`` specification to :func:`.run_cross_validation` will make 
it look like this:

.. code-block:: python

    run_cross_validation(X=X, y=y, data=df, X_types=X_types)


So far, we saw in what form Julearn's :func:`.run_cross_validation` needs to 
and can get input data. However, we want to do machine learning with these data.
In the next section we will focus on basic options to use 
:func:`.run_cross_validation` to make different pipelines in a cross-validation
consistent manner.
