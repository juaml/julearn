.. include:: ../links.inc

.. _model_inspection:

Inspecting Models
=================

Understanding the internals of machine learning models is essential for
interpreting their behavior and gaining insights into their predictions. By
inspecting the parameters and hyperparameters of a trained model, we can
identify the features that have the most significant impact on the model's
output and explore how the model works. This is especially important when using
nested cross-validation workflows, where we train and test the model repeatedly
on different subsets of data with various hyperparameters. By analyzing the
performance of each model across different iterations and hyperparameters, we
can assess the variability across models and identify any patterns that might
help interpret the model's outputs. This information is crucial for deploying
machine learning models in real-world applications where interpretability and
transparency are essential. The ability to inspect the internals of machine
learning models can help us identify the most critical features that influence
the model's predictions, understand how the model works and make informed
decisions about its deployment.

In this context, we will explore how to perform model inspection in julearn.
julearn provides an intuitive suite of tools for model inspection and
interpretation. We will focus on how to inspect models in julearn's nested
cross-validation workflow. Specifically, we will demonstrate how to analyze the
performance and variability of the models across different hyperparameters and
iterations. We will also show how to explore the fitted parameters of the final
model and inspect the impact of individual features on the model's output. With
these techniques, we can gain a better understanding of how the model works and
identify any patterns or anomalies that could affect its performance. This
knowledge can help us deploy models more effectively and interpret their outputs
with greater confidence.

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


We are going to use a fairly simple pipeline, in which we zscore the features
and then apply a support vector classifier to classify species.

.. code-block:: python

    # create model
    pipeline_creator = PipelineCreator(problem_type="classification", apply_to="*")
    pipeline_creator.add("zscore")
    pipeline_creator.add("svm", kernel="linear", C=np.geomspace(1e-2, 1e2, 5))


