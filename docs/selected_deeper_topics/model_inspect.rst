.. include:: ../links.inc

.. _model_inspection:

Inspecting Models
=================

Understanding the internals of machine learning models is essential for
interpreting their behavior and gaining insights into their predictions. By
inspecting the parameters and hyperparameters of a trained model, we can
identify the features that have the most significant impact on the model's
output and explore how the model works. By analyzing the performance of each
model across different iterations and hyperparameters, we can assess the
variability across models and identify any patterns that might help interpret
the model's outputs. The ability to inspect the internals of machine
learning models can help us identify the most critical features that influence
the model's predictions, understand how the model works and make informed
decisions about its deployment.

In this context, we will explore how to perform model inspection in julearn.
Julearn provides an intuitive suite of tools for model inspection and
interpretation. We will focus on how to inspect models in julearn's nested
cross-validation workflow. With these techniques, we can gain a better
understanding of how the model works and identify any patterns or anomalies that
could affect its performance. This knowledge can help us deploy models more
effectively and interpret their outputs with greater confidence.

Let's start by importing some useful utilities:

.. code-block:: python

    from pprint import pprint
    import seaborn as sns
    import numpy as np

    from sklearn.model_selection import RepeatedKFold

    from julearn import run_cross_validation
    from julearn.pipeline import PipelineCreator
    from julearn.utils import configure_logging


Now, let's configure julearn's logger to get some output as the pipeline is
running and get some toy data to play with. In this example, we will use the
penguin dataset, and classify the penguin species based on the continuous
measures in the dataset.

.. code-block:: python

    configure_logging(level="INFO")

    # get some data
    penguins_df = sns.load_dataset("penguins")
    penguins_df = penguins_df.drop(columns=["island", "sex"])
    penguins_df = penguins_df.query("species != 'Chinstrap'").dropna()
    penguins_df["species"] = penguins_df["species"].replace(
	{"Adelie": 0, "Gentoo": 1}
    )
    features = [x for x in penguins_df.columns if x != "species"]


We are going to use a fairly simple pipeline, in which we z-score the features
and then apply a support vector classifier to classify species.

.. code-block:: python

    # create model
    pipeline_creator = PipelineCreator(problem_type="classification", apply_to="*")
    pipeline_creator.add("zscore")
    pipeline_creator.add("svm", kernel="linear", C=np.geomspace(1e-2, 1e2, 5))

Once this is set up, we can simply call julearn's :func:`.run_cross_validation`.
Notice, how we set the :code:`return_inspector` parameter to :code:`True`.
Importantly, we also have to set the :code:`return_estimator` to :code:`"all"`.
This is because julearn's :code:`Inspector` extracts all relevant
information from estimators after the pipeline has been run. The pipeline will
take a few minutes in our example:

.. code-block:: python

    scores, final_model, inspector = run_cross_validation(
	X=features,
	y="species",
	data=penguins_df,
	model=pipeline_creator,
	seed=200,
	cv=RepeatedKFold(n_repeats=10, n_splits=10, random_state=200),
	return_estimator="all",
	return_inspector=True,
    )

After this is done, we can now use the inspector to look at final model parameters,
but also at the parameters of individual models from each fold of the
cross-validation. The final model can be inspected using the :code:`.model`
attribute. For example to get a quick overview over the model parameters, run:

.. code-block:: python

    # remember to actually import pprint as above, or just print out using print   
    pprint(inspector.model.get_params())


This will print out a dictionary containing all the parameters of the final selected
estimator. Similarly, we can also get an overview of the fitted parameters:

.. code-block:: python

    pprint(inspector.model.get_fitted_params())


Again, this will print out quite a lot. What if we just want to look at a specific
parameter? Well, this somewhat depends on the underlying structure and attributes
of the used estimators or transformers, and will likely require some interactive
exploring. But the inspector makes it quite easy to interactively explore your
final model. For example, to see which sample means were used to z-score features
in the final model we can run:

.. code-block:: python
		
    print(inspector.model.get_fitted_params()["zscore__mean_"])


In addition, sometimes it can be very useful to know what predictions were made in
each individual train-test split of the cross-validation. This is where the
:code:`.folds` attribute comes in handy. This attribute has a :code:`.predict()`
method, that makes it very easy to display the predictions made for each sample
in each test fold and in each repeat of the cross-validation. It will display
a DataFrame with each row corresponding to a sample, and each column corresponding
to a repeat of the cross-validation. Simply run:

.. code-block:: python

    print(inspector.folds.predict())

This :code:`.folds` attribute is actually an iterator, that can iterate over
every single fold used in the cross-validation, and it yields an instance of a
:code:`FoldsInspector`, which can then be used to explore each model that was
fitted during cross-validation. For example, we can collect the 'C' parameters
that were selected in each outer fold of our nested cross-validation. That way,
we can assess the amount of variance on that particular parameter across folds:

.. code-block:: python

    c_values = []
    for repeat, fold, fold_inspector in zip(
	scores["repeat"], scores["fold"], inspector.folds
    ):
	print(f"Repeat {repeat}, fold {fold}\n")
	print(fold_inspector.model.get_fitted_params()["svm__model_"])
	c_values.append(
	    fold_inspector.model.get_fitted_params()["svm__model_"].get_params()[
		"C"
	    ]
	)


By printing out the unique values in the :code:`c_values` list, we realise, that
actually there was not much variance across models. In fact, there was only one
parameter value ever selected. This may indicate that this is in fact the optimal
value, or it may indicate that there is a potential problem with our search grid.

.. code-block:: python

    print(set(c_values))

As you can see the inspector provides you with a set of powerful tools to look
at what exactly happened in your pipeline. It may help you better interpret your
models, understand your results, and identify problems if there are any. By
leveraging these tools, you can gain deeper insights, interpret your models
effectively, and address any issues that may arise. Model inspection serves as
a valuable asset in the deployment of machine learning models, ensuring
transparency, interpretability, and reliable decision-making. With julearn's
model inspection capabilities, you can confidently navigate the complexities of
machine learning models and harness their full potential in real-world applications.

