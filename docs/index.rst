.. include:: links.inc
.. julearn documentation master file, created by
   sphinx-quickstart on Thu Oct 29 14:29:33 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

###################################
Welcome to julearn's documentation!
###################################

.. image:: images/julearn_logo_it.png
   :width: 300
   :alt: julearn


... a user-oriented machine-learning library.

What is Julearn?
================

At the Applied Machine Learning (`AML`_) group, as part of the Institute of 
Neuroscience and Medicine - Brain and Behaviour (`INM-7`_), we thought that
using ML in research could be simpler. 

In the same way as `seaborn`_ provides an abstraction of `matplotlib`_'s
functionality aiming for powerful data visualization with minor coding, we 
built julearn on top of `scikit-learn`_.

Julearn is a library that provides users with the possibility of easy 
testing ML models directly from `pandas`_ dataframes, while keeping the
flexibiliy of using `scikit-learn`_'s models.

To get started with Julearn just keep reading here. Additionally You can 
check out our `video tutorial`_.

Why Julearn?
------------

Why not just using `scikit-learn`? Julearn offers **three essential benefits**:

1. You can do machine learning with **less amount of code** than in
   `scikit-learn`
2. Julearn helps you to build and evaluate pipelines in an easy way and thereby
   helps you **avoid data leakage**!
3. It offers you nice **additional functionality**:
   
   * Easy to implement **confound removal**: Julearn offers you a simple way
     to remove confounds from your data in a cross-validated way.
   * Data **typing**: Julearn provides a system to specify **data types** for
     your features, and then provides you with the possibility to 
     filter and transform your data according to these types.
   * Model **inspection**: Julearn provides you with a simple way to **inspect**
     your models and pipelines, and thereby helps you to understand what is
     going on in your pipeline.
   * Model **comparison**: Julearn provides out-of-the-box interactive
     **visualizations** and **statistics** to compare your models.

Table of Contents
-----------------

.. toctree::
   :maxdepth: 2
   :numbered: 2

   getting_started

   what_really_need_know/index.rst

   selected_deeper_topics/index.rst

   available_pipeline_steps.rst

   examples.rst

   api/index.rst
   configuration
   contributing
   maintaining
   faq
   whats_new


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

