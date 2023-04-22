.. include:: ../links.inc

.. _need_to_know:

What you really need to know
============================

The backbone of Julearn is the function :func:`.run_cross_validation`, which
allows you do all the magic. All important information needed for your 
machine learning workflow goes into this function, specified via its parameters.

But why is basically everything based on one `cross_validation` function? Well,
because doing proper cross-validation is the backbone of machine learning.

Why cross validation?
---------------------

One of the 

Ressources to cite:
From https://www.sciencedirect.com/science/article/pii/S105381191630595X:
cross-validation, the standard tool to measure predictive power and tune parameters in decoding.
Read this paper particularly the section
"A primer on cross-validation" to understand ... 
... important concepts in cross validation for decoding from brain images.

Mostly:
https://www.nature.com/articles/s41746-022-00592-y
very summarized why CV is good in section:
"Improper evaluation procedures and leakage"


Maybe:
https://www.sciencedirect.com/science/article/pii/S1053811917305311

scikitlearn:
https://scikit-learn.org/stable/modules/cross_validation.html


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
   :caption: Table of Contents
   
   data.rst
   pipeline.rst
   model_evaluation.rst

If you are just interested in seeing all parameters of :func:`.run_cross_validation`, 
click on the function link to have a look at all its parameters in the `api`.

If you are already familiar with how to set up a basic workflow using Julearn 
and want to do more fancy stuff, go to :ref:`selected_deeper_topics` to pick 
and select specific topics.


