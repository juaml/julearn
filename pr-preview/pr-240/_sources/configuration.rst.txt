.. include:: links.inc

.. _configuration:

Configuring ``julearn``
=======================

While ``julearn`` is meant to be a user-friendly tool, this also comes with a
cost. For example, in order to provide the user with information as well as to
be able to detect potential errors, we have implemented several checks. These
checks, however, might yield high computational costs. Therefore, we have
implemented a global configuration module in ``julearn`` that allows to set
flags to enable or disable certain extra functionality. This module is called
``julearn.config`` and it has a single function called ``set_config``
that given a configuration flag name and a value, it sets the flag to the given
value.

Here you can find the comprehensive list of flags that can be set:

.. list-table::
   :widths: auto
   :header-rows: 1

   * - Flag
     - Description
     - Potential problem(s)
   * - ``disable_x_check``
     - | Disable checking for unmatched column names in ``X``.
       | If set to ``True``, any element in ``X`` that is not present in the
       | dataframe will not result in an error.
     - | The user might think that a certain feature is used in the model when
       | it is not.
   * - ``disable_xtypes_check``
     - | Disable checking for missing/present ``X_types`` in the ``X`` parameter
       | of the :func:`.run_cross_validation` method.
       | If set to ``True``, the ``X_types`` parameter will not be checked for
       | consistency with the ``X`` parameter, including undefined columns in
       | ``X``, missing types in ``X_types`` or duplicated columns in
       | ``X_types``.
     - | The user might think that a certain feature is considered in the model
       | when it is not.
   * - ``disable_x_verbose``
     - | Disable printing the list of expanded column names in ``X``.
       | If set to ``True``, the list of column names will not be printed.
     - The user will not see the expanded column names in ``X``.
   * - ``disable_xtypes_verbose``
     - | Disable printing the list of expanded column names in ``X_types``.
       | If set to ``True``, the list of types of X will not be printed.
     - The user will not see the expanded ``X_types`` column names.
