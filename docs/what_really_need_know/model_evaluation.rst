.. .. include:: ../links.inc

.. .. _model_evaluation_usage:

.. Model evaluation
.. ================

.. Content to treat: cv, scoring -> external skl input, access train/test scores, t-test for model comparison

.. .. include:: links.inc

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