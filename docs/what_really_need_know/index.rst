.. .. include:: ../links.inc

.. What you really need to know
.. ============================

.. The backbone of Julearn is the function :func:`.run_cross_validation`, which
.. allows you do all the magic. All important information needed for your 
.. machine learning workflow goes into this function, specified via its parameters.

.. The essence of :func:`.run_cross_validation`
.. --------------------------------------------

.. :func:`.run_cross_validation` uses your **specified parameters** to train a model 
.. accordingly and most importantly does all specified steps in a cross-validation 
.. consistent manner. This helps to avoid data leakage.

.. Among others, the parameters passed to :func:`.run_cross_validation` include 
.. the specification of ...

.. 1. ... your data and feature types (see :ref:`data_usage`)
.. 2. ... your pipeline steps, like the learning algorithm or preprocessing steps 
..    to use (see :ref:`pipeline_usage`)
.. 3. ... how the model should be evaluated, like cross validation scheme or the 
..    scoring to be used for evaluation (see :ref:`model_evaluation_usage`) 

.. After training the model either only the model's scores from the cross 
.. validation or the scores and the model(s) of the performed cross validation are 
.. **returned** (depending on your exact specifications).

.. Basics
.. ------

.. To see what is needed to setup a basic workflow using :func:`.run_cross_validation` 
.. follow along this chapter where you can find out more about the data input, 
.. a basic pipeline and basic model evaluation:

.. .. toctree::
..    :maxdepth: 2
..    :caption: Table of Contents
..    :numbered:

..    data.rst
..    pipeline.rst
..    model_evaluation.rst

.. If you are just interested in seeing all parameters of :func:`.run_cross_validation`, 
.. click on the function to have a look at all its parameters in the `api`.

.. If you are already familiar with how to set up a basic workflow using Julearn 
.. and want to do more fancy stuff, go to :ref:`selected_deeper_topics` to pick 
.. and select specific topics.


