# Authors: Vera Komeyer <v.komeyer@fz-juelich.de>
#          Fede Raimondo <f.raimondo@fz-juelich.de>
# License: AGPL

"""
Data
====

Data input to :func:`.run_cross_validation`
-------------------------------------------

``julearn`` deals with data in the form of ``pandas.DataFrames``. This is the
kind of data structure that the :func:`.run_cross_validation` uses to input the
data and output some of the results.

The input DataFrame must contain the features and the target or label. This
will be communicated to :func:`.run_cross_validation` by specifying the
following parameters:

- ``data``: Name of the DataFrame containing the features and the target or
   label.
- ``X``: List of string containing the column names of the features.
- ``y``: String containing the name of the column with the target or label.

For example, using the well known ``iris`` dataset, we can specify the data
input as follows:

First, we load the data into a ``pandas.DataFrame`` called ``df`` and specify
``X`` and ``y``:
"""

from seaborn import load_dataset

df = load_dataset("iris")

##############################################################################
# Let's inspect what our dataframe looks like.

df.head()

##############################################################################
# Given this data, we can now specify the ``X`` and ``y`` parameters:

X = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
y = "species"

##############################################################################
# ``julearn``'s :func:`.run_cross_validation` function so far would look like
# this:
#
# .. code-block:: python
#
#    run_cross_validation(X=X, y=y, data=df)
#
# This is not yet very useful to do machine learning, but we will come to it
# step by step.

##############################################################################
# Giving ``types`` to features
# ----------------------------
#
# A nice add-on that ``julearn`` offers is the capacity to specify colum-based
# types for the features. This comes in handy if within the pipeline, one
# wants to manipulate only certain columns.
#
# To specify column types, we must provide a dictionary with the column types
# as keys and the column names as values. The type can be anything, but it is
# recommended to use a string that is meaningful to you.
#
# .. important::
#    Every column can only have **one type**!
#
#
# In the case of the ``iris dataset``, we could specify the type of the columns
# related to the ``sepal`` and ``petal`` information as ``"sepal"`` and
# ``"petal"`` respectively.

X_types = {
    "petal": ["petal_length", "petal_width"],
    "sepal": ["sepal_length", "sepal_width"],
}

##############################################################################
# Importantly, ``julearn`` also allows to specify the column names as regular
# expressions. This comes in handy when we are dealing with hundreds or
# thousands of features and we do not want to specify all the names by hand.
# For example, we could specify the type of the ``sepal`` columns
# as follows:

X_types = {
    "petal": ["petal.*"],
    "sepal": ["sepal.*"],
}


##############################################################################
# Adding an ``X_types`` specification to :func:`.run_cross_validation` will
# make it look like this:
#
# .. code-block:: python
#
#     run_cross_validation(X=X, y=y, data=df, X_types=X_types)
#

##############################################################################
# .. important::
#    If no ``X_types`` is specified, all the columns will be considered as
#    ``"continuous"`` and a warning will be raised.
#
#
# Until now we saw how to parametrize :func:`.run_cross_validation` in terms
# of the input data. In the next section we will see how to specify the output.
# In the next section we will focus on basic options to use
# :func:`.run_cross_validation` to evaluate different pipelines in a
# cross-validation consistent manner.
#
# Advanced uses cases regarding ``X_types`` selective processing are covered in
# :ref:`apply_to_feature_types`.
