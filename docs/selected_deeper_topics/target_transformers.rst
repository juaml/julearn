
Applying preprocessing to the target
------------------------------------

What we covered so far is how to apply preprocessing to the features and train 
a model in a cv-conistent manner by building a pipeline.
However, sometimes one wants to apply preprocessing to the target. For example,
when having a regression-task (continous target variable), one might want to 
predict the z-scored target.
This can be achieved by using a :class:`.TargetPipelineCreator` 
as one step in the general pipeline.

We first create a :class:`.TargetPipelineCreator`:

.. code-block:: python

    target_creator = TargetPipelineCreator()
    target_creator.add("zscore")

Next, we create the general pipeline using a :class:`.PipelineCreator`. We pass
the ``target_creator`` as one step of the pipeline and specify that it should 
only be applied to the ``target``, which makes it clear for Julearn to only 
apply it to ``y``:

.. code-block:: python

    creator = PipelineCreator(problem_type="regression")
    creator.add(target_creator, apply_to="target")
    creator.add("svm")

This ``creator`` can then be passed to :func:`.run_cross_validation` as usual:

.. code-block:: python

    run_cross_validation(
        X=X,
        y=y,
        data=df, 
        model=creator
    )


All transformers in (:ref:`available_transformers`) can be used for both, 
feature and target transformations. However, features transformations can be 
directly specified as step in the :class:`.PipelineCreator`, while target 
transformations have to be specified using the :class:`.TargetPipelineCreator`, 
which is then passed to the overall :class:`.PipelineCreator` as an extra step.
