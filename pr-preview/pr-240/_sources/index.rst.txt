.. julearn documentation master file, created by
   sphinx-quickstart on Thu Oct 29 14:29:33 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. include:: links.inc

Welcome to julearn's documentation!
===================================

.. image:: images/julearn_logo_it.png
   :width: 300px
   :alt: julearn logo

... a user-oriented machine-learning library.

What is ``julearn``?
--------------------

At the Applied Machine Learning (`AML`_) group, as part of the Institute of
Neuroscience and Medicine - Brain and Behaviour (`INM-7`_), we thought that
using ML in research could be simpler.

In the same way as `seaborn`_ provides an abstraction of `matplotlib`_'s
functionality aiming for powerful data visualization with minor coding, we
built ``julearn`` on top of `scikit-learn`_.

``julearn`` is a library that provides users with the possibility of easy
testing ML models directly from `pandas`_ DataFrames, while keeping the
flexibility of using `scikit-learn`_'s models.

To get started with ``julearn`` just keep reading here. Additionally you can
check out our `video tutorial`_.

Why ``julearn``?
----------------

Why not just use ``scikit-learn``? ``julearn`` offers **three essential benefits**:

#. You can do machine learning with **less amount of code** than in
   ``scikit-learn``.
#. ``julearn`` helps you build and evaluate pipelines in an easy way and thereby
   helps you **avoid data leakage**!
#. It offers you nice **additional functionality**:

   * Easy to implement **confound removal**: ``julearn`` offers you a simple way
     to remove confounds from your data in a cross-validated way.
   * Data **typing**: ``julearn`` provides a system to specify **data types**
     for your features, and then provides you with the possibility to filter and
     transform your data according to these types.
   * Model **inspection**: ``julearn`` provides you with a simple way to
     **inspect** your models and pipelines, and thereby helps you to understand
     what is going on in your pipeline.
   * Model **comparison**: ``julearn`` provides out-of-the-box interactive
     **visualizations** and **statistics** to compare your models.


Table of Contents
=================

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
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


Indices and tables
==================

If you use julearn in a scientific publication, please use the following 
reference

    Hamdan, Sami, Shammi More, Leonard Sasse, Vera Komeyer, 
    Kaustubh R. Patil, and Federico Raimondo. ‘Julearn: 
    An Easy-to-Use Library for Leakage-Free Evaluation and Inspection of 
    ML Models’. arXiv, 19 October 2023. 
    https://doi.org/10.48550/arXiv.2310.12568.

Since julearn is also heavily reliant on scikit-learn, plase also cite 
them: https://scikit-learn.org/stable/about.html#citing-scikit-learn
