.. include:: links.inc

Advanced Topics
===============

The following sections are advanced topic which do not need to be read
for a lot of usecases, but still provide some context for those who want it.

Column Type System
******************

Context
^^^^^^^
To be able to discriminate between different types of variables Julearn
uses a Column Type System. This system currently distinguishes between
continuous variables/features, categorical variables/features and confounds.

.. note::
  On most levels of Julearn this Column Type System is only used internally.
  Therefore, users do not have to work with it directly.
  For example, by providing the confounds and categorical variables to the
  :class:`.ExtendedDataFramePipeline` it has all the information needed to
  apply the Column Type System internally without any further input or changes
  to the `pandas.DataFrame`.

How it works
^^^^^^^^^^^^
Every `pandas.DataFrame`_ column has a column name.
Inside of Julearn we add another string containing the type of the column
separated by our delimiter: ``'__:type:__'`` to the original column names.
For example:

  * We have the original columns :

    - ``'Intelligence'``
    - ``'Age'``
    - ``'LikesEvoks'``

  * We know:

    - Intelligence is a **continuous** variable
    - Age is a **confound**
    - LikesEvoks is a **categorical** variable.
      Either someone likes Evoks or not.
  * Inside of Julearn's Column Type System we can provide this information
    by changing the column names to:

      - ``'Intelligence__:type:__continuous'``
      - ``'Age__:type:__confound'``
      - ``'LikesEvoks__:type:__categorical'``