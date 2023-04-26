.. include:: ../links.inc

.. _need_to_know:

############################
What you really need to know
############################


The backbone of Julearn is the function :func:`.run_cross_validation`, which
allows you to do all the magic. All important information needed for your 
machine learning workflow goes into this function, specified via its parameters.

But why is basically everything based on one `cross_validation` function? Well,
because doing proper cross-validation is the backbone of machine learning.

.. _why_cv:

Why cross validation?
---------------------

One of the core important concepts in machine learning is cross validation. The 
core idea is that we want to train (fit) a model on a subset of our data and 
evaluate it on a different subset of our data to see how well the trained 
model generalizes to unseen data. Thus, we need to have
separate data for fitting and predicting. As data is a valuable ressource in 
machine learning, cross validation comes in handy as a technique to split the 
data into a training and validation data set using multiple folds.
The training data set is used to 
fit the pipeline, while the validation data set is used to predict the data. 
The predictions are then compared to the true values of the validation data set, 
obtaining an estimation of the prediction performance of the model. This is 
done in a repeated manner for all folds and the overview of the scores from all
folds give a good estimation of the model's generalization performance. To read 
more about cross validation, its functionality and usage and why it is such an
important concept in machine learning, you can have a look at these 
[#1]_ [#2]_ [#3]_ [#4]_ resources.

The essence of :func:`.run_cross_validation`
--------------------------------------------

:func:`.run_cross_validation` uses your **specified parameters** to train a model 
accordingly and most importantly does all specified steps in a cross-validation 
consistent manner. This helps to avoid data leakage.

Among others, the parameters passed to :func:`.run_cross_validation` include 
the specification of ...

1. ... your data and feature types (see :ref:`data_usage`)
2. ... your pipeline steps, like the learning algorithm or preprocessing steps 
   to use (see :ref:`pipeline_usage`)
3. ... how the model should be evaluated, like cross validation scheme or the 
   scoring to be used for evaluation (see :ref:`model_evaluation_usage`) 

After training the model, :func:`.run_cross_validation` either **returns** 
only the model's scores from the cross validation or both, the scores and 
the model(s) of the performed cross validation are 
(depending on your exact specifications).

Get the basics running
----------------------

To see what is needed to setup a basic workflow using :func:`.run_cross_validation`, 
follow along this chapter where you can find out more about the data input, 
a basic pipeline and basic model evaluation:

.. toctree::
   :maxdepth: 2

   cross_validation.rst
   data.rst
   pipeline.rst
   model_evaluation.rst

If you are just interested in seeing all parameters of :func:`.run_cross_validation`, 
click on the function link to have a look at all its parameters in the `api`.

If you are already familiar with how to set up a basic workflow using Julearn 
and want to do more fancy stuff, go to :ref:`selected_deeper_topics` to pick 
and select specific topics.


.. topic:: References:

      .. [#1] https://www.sciencedirect.com/science/article/pii/S105381191630595X

      .. [#2] https://www.nature.com/articles/s41746-022-00592-y

      .. [#3] https://www.sciencedirect.com/science/article/pii/S1053811917305311

      .. [#4] https://scikit-learn.org/stable/modules/cross_validation.html