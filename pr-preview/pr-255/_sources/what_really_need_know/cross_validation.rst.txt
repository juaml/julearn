.. _why_cv:

Why cross validation?
=====================

Cross-validation - The fundamentals
-----------------------------------

   "The fundamental goal of machine learning is to generalize beyond the
   examples in the training set. This is because, no matter how much data we
   have, it is very unlikely that we will see those exact examples again at
   test time.”

   -- Domingos, 2012, A Few Useful Things to Know about Machine Learning

This means that in order to evaluate if a model is *successful* in learning,
we need to evaluate if it is able to predict new data. Thus, we need to have
separate data for learning and testing. At the same time, data is a valuable
resource in machine learning and one wants to use it as efficeint as possible.

To solve this, we use *cross validation*. The core idea is that we want to
train (also named *fit*) a model on a subset of our data and evaluate it on a
different subset of our data to see how well the trained model generalizes to
unseen data. The training set is used to fit a model
(see :ref:`pipeline_usage`), while the validation set is used to predict the
data labels. The predictions are then compared to the true labels of the
validation dataset, obtaining an estimation of the prediction performance of
the model.

There are several ways to split the data into training and validation sets.
The most common way is to split the data into two parts, the training set and
the validation set. However, this approach has the disadvantage that the
validation set is only used once and thus, the estimation of the prediction
performance is based on a small number of data points. This can lead to
unstable results. To overcome this problem, cross validation is used. In
cross validation, the data is split into *k* *folds* (splits). Then, *k*
models are trained, each time using a different fold as the validation set and
the remaining folds as the training set. This procedure can be repeated several
times, each time with a different split of the data into folds.

To read more about cross validation, its functionality and usage and why it is
such an important concept in machine learning, you can have a look at these
[#1]_ [#2]_ [#3]_ [#4]_ resources.


The essence of :func:`.run_cross_validation`
--------------------------------------------

Building pipelines (see :ref:`pipeline_usage`) within a (nested)
cross-validation scheme, without accidentally leaking some information between
steps can quickly become complicated and errors are often not-obvious to
detect. ``julearn``'s :func:`.run_cross_validation` provides a simple and
straightforward way to do cross-validation less prone to such accidental
mistakes and more transparent for debugging. The user only needs to specify
the model to be used, the data to be used and the evaluation scheme to be used.
``julearn`` then builds the pipeline, splits the data into training and
validation sets accordingly, and most importantly, does all specified steps in a
cross-validation consistent manner.

The main parameters needed for :func:`.run_cross_validation` include the
specification of:

#. ``data``: the data, including features, labels and feature types
   (see :ref:`data_usage`)
#. ``model``: the model to evaluate, including the data transformation steps
   and the learning algorithm to use (see :ref:`pipeline_usage`).
#. ``model evaluation``: how the model performance should be estimated,
   like the cross validation scheme or the metrics to be computed
   (see :ref:`model_evaluation_usage`)

The  :func:`.run_cross_validation` function will then output the DataFrame with
the fold-wise metrics, which can then be used to visualize and evaluate the
estimation of the models' performance.

Additional parameters can be used to control the output of the function, in
order to provide mechanisms for model inspection and debugging.

See :ref:`model_evaluation_usage` for further details on the model evaluation.


.. topic:: References:

      .. [#1] https://www.sciencedirect.com/science/article/pii/S105381191630595X

      .. [#2] https://www.nature.com/articles/s41746-022-00592-y

      .. [#3] https://www.sciencedirect.com/science/article/pii/S1053811917305311

      .. [#4] https://scikit-learn.org/stable/modules/cross_validation.html
