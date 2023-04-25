.. include:: ../links.inc

.. _model_evaluation_usage:

Model evaluation
================

Content to treat: 
  model and score as output of run_cv -> inspect some output from the examples in pipeline.rst
  cv, + return estimator=final
  scoring -> external skl input, access train/test scores, t-test for model comparison
  inspect preprocess??


.. The returned pipeline
.. ----------------------

.. The :func:`.run_cross_validation` uses all the information mentioned above
.. to create one ExtendedDataFramePipeline which is then used for
.. cross_validation. Additionally, it can return the fitted pipeline for other
.. application. E.g. you could want to test the pipeline on one additional
.. test set. But how can you do that?

.. Returning the (extended) pipeline
.. ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. There are multiple options which you can use to return the pipeline(s).
.. For all of them you have to set the `return_estimator`.
.. These are the possible options:

..   * None: Does not return any estimator
..   * ``'final'``: Return the estimator fitted on all the data.
..   * ``'cv'``: Return the all the estimator from each CV split, fitted on the
..     training data.
..   * ``'all'``: Return all the estimators (final and cv).

.. These returned estimators are always ExtendedDataFramePipeline 
.. objects.Therefore, the next section will discuss how you can use
.. a returned estimator.

.. ExtendedDataFramePipeline
.. ^^^^^^^^^^^^^^^^^^^^^^^^^
.. The ExtendedDataFramePipeline has the same basic functionality as
.. all scikit-learn pipelines or estimators, but also has some caveats.

.. Where ExtendedDataFramePipeline behave as usual
.. ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. The following methods work as in sklearn:

..   * ``.fit()``
..   * ``.predict()``
..   * ``.score()``
..   * ``.predict_proba()``

.. Caveats ExtendedDataFramePipeline
.. ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. In contrast to scikit-learn pipelines ExtendedDataFramePipeline
.. can change the ground truth (transform the target).
.. This means that any any function which uses sklearn scorer functions instead of
.. calling ``.score()`` on the ExtendedDataFramePipeline can give you
.. the wrong output without **any warning**.
.. For example `cross_validate` function of sklearn when using another scorer.

.. If you want to use such functions, you can follow this example (#TODO) which
.. shows how to use julearns ``extended_scorer`` instead


.. Additional functionality
.. ^^^^^^^^^^^^^^^^^^^^^^^^
.. Furthermore, ExtendedDataFramePipeline  have the following
.. added methods:

..   * ``preprocess``: a method to apply preprocessing steps of the pipeline to
..     some data. Furthermore, the ``until`` argument can be used to
..     only preprocess up to a specific transformer.




.. Scoring
.. =======

.. On top of scikit-learn `scoring`_ parameter options, julearn extends the 
.. functionality with more internal scorers and the possibility to define custom
.. scorers.

.. Internal Scorers
.. ****************

.. .. list-table::
..    :widths: 30 80
..    :header-rows: 1

..    * - Name (str)
..      - Description
..    * - ``r2_corr``
..      - Pearson product-moment correlation coefficient (squared), as computed by
..        `numpy.corrcoef`_


.. Custom Scorers
.. **************

.. julearn allows the user to define any function and use it as a scorer in the 
.. same way scikit-learn or julearn internal scorers work.

.. In the example (TODO: place link) you can see how to make use of this functionality.