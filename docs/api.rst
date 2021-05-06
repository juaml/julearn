Reference
=========
.. include:: links.inc


Main API functions
^^^^^^^^^^^^^^^^^^

.. automodule:: julearn.api
   :members:

Model functions
^^^^^^^^^^^^^^^

.. autofunction:: julearn.estimators.list_models
.. autofunction:: julearn.estimators.get_model

Transformer functions
^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: julearn.transformers.list_transformers
.. autofunction:: julearn.transformers.get_transformer

=======

Logging
^^^^^^^
.. autofunction:: julearn.utils.configure_logging
.. autofunction:: julearn.utils.warn
.. autofunction:: julearn.utils.raise_error

=======

Cross-Validation
^^^^^^^^^^^^^^^^

.. autoclass:: julearn.model_selection.StratifiedBootstrap
   :members: 

.. autoclass:: julearn.model_selection.StratifiedGroupsKFold
   :members: 

.. autoclass:: julearn.model_selection.RepeatedStratifiedGroupsKFold
   :members: 

=======

Classes
^^^^^^^

.. autoclass:: julearn.transformers.confounds.DataFrameConfoundRemover
   :members: 
.. autoclass:: julearn.transformers.confounds.TargetConfoundRemover
   :members: 
.. autoclass:: julearn.pipeline.ExtendedDataFramePipeline
   :members: