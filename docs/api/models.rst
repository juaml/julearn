Models
======

.. automodule:: julearn.models
   :no-members:
   :no-inherited-members:

Functions
---------

.. currentmodule:: julearn.models

.. autosummary::
   :toctree: generated/
   :template: function.rst

    list_models
    get_model
    register_model
    reset_model_register

Julearn custom models
---------------------

This is a list of models implemented by Julearn that are not simple wrappers
around existing models in other libraries but rather variants of existing
models or novel models.

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: class.rst

   xgb_cvearlystopping.XGBClassifierCVEarlyStopping
   xgb_cvearlystopping.XGBRegressorCVEarlyStopping

Dynamic Selection (DESLib)
==========================

.. warning::
   Dynamic Selection is temporarily unavailable: the latest `DESlib`_
   release (0.3.7) is not compatible with scikit-learn >= 1.6. It will be
   available again once a newer DESlib version is released.

.. automodule:: julearn.models.dynamic
   :no-members:
   :no-inherited-members:

Classes
-------

.. currentmodule:: julearn.models.dynamic

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: class.rst

    DynamicSelection
