.. include:: links.inc

Scoring
=======

On top of scikit-learn `scoring`_ parameter options, julearn extends the 
functionality with more internal scorers and the possibility to define custom
scorers.

Internal Scorers
****************

.. list-table::
   :widths: 30 80
   :header-rows: 1

   * - Name (str)
     - Description
   * - ``r2_corr``
     - Pearson product-moment correlation coefficient (squared), as computed by
       `numpy.corrcoef`_


Custom Scorers
**************

julearn allows the user to define any function and use it as a scorer in the 
same way scikit-learn or julearn internal scorers work.

In the example 
:ref:`sphx_glr_auto_examples_advanced_run_custom_scorers_regression.py`
you can see how to make use of this functionality.