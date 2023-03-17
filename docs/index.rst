.. include:: links.inc
.. julearn documentation master file, created by
   sphinx-quickstart on Thu Oct 29 14:29:33 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to julearn's documentation!
===================================

.. image:: images/julearn_logo_it.png
   :width: 300
   :alt: julearn


... a user-oriented machine-learning library.

What is Julearn?
----------------

At the Applied Machine Learning (`AML`_) group, as part of the Institute of 
Neuroscience and Medicine - Brain and Behaviour (`INM-7`_), we thought that
using ML in research could be simpler. 

In the same way as `seaborn`_ provides an abstraction of `matplotlib`_'s
functionality aiming for powerful data visualization with minor coding, we 
built julearn on top of `scikit-learn`_.

Julearn is a library that provides users with the possibility of easy 
testing ML models directly from `pandas`_ dataframes, while keeping the
flexibiliy of using `scikit-learn`_'s models.

You can also check out our `video tutorial`_.

Why Julearn?
------------

Why not just using `scikit-learn`? Julearn offers **three essential benefits**:

1. You can do machine learning with **less amount of code** than in
   `scikit-learn`
2. Julearn helps you to build pipelines in an easy way and thereby supports you
   to **avoid data leakage**
3. It offers you nice **additional functionality**:
   
   * Easy to implement **confound removal**  # TODO Link to example
   * Data type **typing**  # TODO Link to example, name in a bette way

Julearn's backbone - :func:`.run_cross_validation`
--------------------------------------------------

The backbone of Julearn is the function :func:`.run_cross_validation`, which let's you 
do all the magic.  # TODO Add more?, link to most essential part in docu on runcv

Table of Contents
-----------------

.. toctree::
   :maxdepth: 2
   :numbered:

   getting_started
   what_really_need_know/index.rst
   what_really_need_know/data.rst
   what_really_need_know/pipeline.rst
   what_really_need_know/model_evaluation.rst
   selected_deeper_topics/index.rst
   selected_deeper_topics/confound_removal.rst
   selected_deeper_topics/HPT.rst
   selected_deeper_topics/model_inspect.rst
   selected_deeper_topics/stacked_models.rst
   selected_deeper_topics/CBPM.rst

   input
   pipeline
   scoring
   steps
   hyperparameters
   auto_examples/basic/index.rst
   auto_examples/advanced/index.rst
   api
   contributing
   maintaining
   whats_new


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

