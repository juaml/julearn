.. include:: ../links.inc

.. _pipeline_usage:

Pipeline
========

So far we know how to pass input data to :func:`.run_cross_validation` 
(see :ref:`data_usage`). Now, we will have a look at the implementation of 
basic machine learning pipelines.

A machine learning pipeline is a process of automating the workflow of a 
machine learning task. It can be thought of as a combination of pipes and 
filters. In the beginning the raw data is put into this pipeline. Different 
fitlers within the pipeline modify the data and the output of the machine 
learning pipeline can be (among others) the machine learning model, 
model parameters or prediction outputs.

Julearn aims to provide a user-friendly way to apply complex machine learning
pipelines. The :func:`.run_cross_validation` function makes it easy to specify 
and costumize the pipeline. We first have a look at the most basic pipeline, 
only consisting of a machine learning algorithm. Then we will make the 
pipeline incrementally more complex.

.. _basic_cv:

Basic cross validation with Julearn
-----------------------------------

The core of doing machine learning is the learning algorithm used. 
:func:`.run_cross_validation` can be told which algorithm to use by 
specifying its ``model``-parameter . :func:`.run_cross_validation` 
basically can take all scikit-learn compatible learning algorithms as input
to its ``model`` parameter. In particular, click on
:ref:`available_models` to see the models that can be used.

A very prominent and often used learning algorithm in machine learning is a 
support vector machine (SVM) [#1]_. We will use this algorithm in the following
together with the example 'iris' data. The model is given to 
:func:`.run_cross_validation` by its name as a string as indicated in the 
column 'name' in :ref:`available_models`.

.. code-block:: python

  run_cross_validation(
      X=X, y=y, data=df, 
      model = "svm",
  )

This code, however, will still give you an error. This is because in machine 
learning, one can distinguish between regression problems, when predicting a 
continous outcome and classification problems for discrete class label 
predictions. A SVM can be used for both, regression or classification problems.
Therefore, :func:`.run_cross_validation` additionally needs to know, which 
problem type one is interested in. This is done by specifying the parameter 
``problem_type``. The possible values are ``classification`` and 
``regression`` and importantly, there is no default specified, so you have to 
explicitely specify it. In the example we are interested in
predicting the species (see ``y`` in :ref:`data_usage`), i.e. a discrete class 
label. Therefore, the ``problem_type`` is `classification`. 

.. code-block:: python

  run_cross_validation(
      X=X, y=y, data=df, 
      model = "svm",
      problem_type = "classification",
  )

Et voilà, your first machine learning pipeline is ready to go.

(Feature) preprocessing
-----------------------
There are cases in which the input data, and in particular the features, 
should be transformed before passing them to the learning algorithm. One 
scenario can be that certain learning algorithms need the features in a 
specific form, for example in standardized form so that the data resemble a 
normal distribution. This can be achieved by first z-scoring (or standard 
scaling) the features (ref z-score). 

Importantly, in a machine learning workflow all transformations done to the 
data have to be done in a cv-consistent way. That means, that one should do 
steps like feature preprocessing on the training data of each respective cross 
validation fold and then only apply the parameters of the transformation to the 
validation data of the respective fold. One should **never** do preprocessing on 
the entire dataset and then do cross validation on the already preprocessed 
features (or more generally transformed data) because this leads to leakage of 
information from the training into the validation data. This is exactly where
:func:`.run_cross_validation` comes in handy, because you can simply add your
wished preprocessing step (:ref:`available_feature_preprocessing`) and it 
takes care of doing the respective transformations in a cv-consistent manner.   


.. code-block:: python

  run_cross_validation(
      X=X, y=y, data=df, 
      preprocess="zscore",
      model = "svm"
  )

.. note::
  Learning algorithms (what we specified in the `model`-parameter),
  are estimators. Preprocessing steps however, are usually transformers, because 
  they transform the input data in a certain way. Therefore the parameter 
  description in the api of :func:`.run_cross_validation`, 
  defines valid input for the `preprocess`-parameter as `TransformerLike`.

  preprocess : str, TransformerLike or list or PipelineCreator | None
          Transformer to apply to the features. If string, use one of the
          available transformers. If list, each element can be a string or
          scikit-learn compatible transformer. If None (default), no
          transformation is applied.

But what if we want to add more pre-processing steps? 
A common scenario can be, that there is 
many features available, so that one wants to first reduce the dimensionality 
of the features before passing them to the learning algorithm. A commonly used 
approach is a principal component analysis (PCA) (ref pca). If we nevertheless 
want to keep our previously applied z-scoring, we can simply add the PCA as 
another preprocessing step as follows:

.. code-block:: python

  run_cross_validation(
      X=X, y=y, data=df, 
      preprocess=["zscore","pca"],
      model = "svm"
  )

This is nice, but with more steps added to the pipeline this can become 
intransparent. Therefore, Julearn provides a ``PipelineCreator`` which helps 
keeping things neat.

.. _pipeline_creator:

Pipeline specification made easy with the ``PipelineCreator``
-------------------------------------------------------------
The ``PipelineCreator`` is a class that helps to create a pipeline by 
recursively converting given parameters to pipelines. On a first read that 
might sound cryptic, but the usage is straightforward.

1. **Transformer Pipeline**: (feature) preprocessing with the pipeline creator
   
   Let's start by doing the exact same as in the code example above, but now we 
   will use the ``PipelineCreator`` to specify our pre-processing steps.

   .. code-block:: python

    from julearn.pipeline import PipelineCreator

    creator = (PipelineCreator()
            .add("zscore")
            .add("pca")
            )
    
   This creator can then simply be passed to the ``preprocess``-parameter of 
   :func:`.run_cross_validation`, just as we did it before with the list 
   specification of pre-processing steps.

   .. code-block:: python

    run_cross_validation(
      X=X, y=y, data=df, 
      preprocess=creator,
      model = "svm"
    )

Ultimately, both the pre-processing steps and the passed learning-algorithm 
are subsequent steps of the pipeline. Following this logic, we can also add 
the specified learning algorithm to the pipeline creator.

2. **Complete Pipeline**: Create the entire model with the pipeline creator
   
   .. code-block:: python

    creator = (PipelineCreator()
            .add("zscore")
            .add("pca")
            .add("svm")
            )
    
   We again pass the ``PipelineCreator`` to :func:`.run_cross_validation` but 
   this time to the ``model``-parameter and not the ``preprocess``-parameter. 
   Nevertheless, all specified pre-processing steps will be executed.

   .. code-block:: python

    run_cross_validation(
      X=X, y=y, data=df, 
      model = creator
    )

.. note::
  If the argument passed to the ``model``-parameter is a PipelineCreator, 
  the preprocess parameter should be None. Otherwise, an error will be raised. 
  This can be done by just not specifying it as the default is None.

Awesome! We covered how to create a basic machine learning pipeline and 
even added multiple feature pre-preprocessing steps. 

Let's jump to the next important aspect in a machine learning 
pipeline: **Hyperparameters**. We here cover the basics of specifying 
hyperparameters. If you want to know more about tuning (or optimizing) 
hyperparameters, please have a look at :ref:`hp_tuning`.

How to specify hyperparameters
------------------------------
If you are new to machine learning, the section heading might confuse you: 
Parameters, hyperparameters - aren't we doing machine learning, so shouldn't 
the model learn all our parameters? Well, yes and no. Yes, it should learn 
parameters. However, hyperparameters and parameters are two different things. 

that are not learned from the data but are...

A **model parameter** is a variable that is internal to the
algorithm and we want to learn or estimate its value from the data, which in 
turn means that they are not set manually. They are required by the model and 
are often saved as part of the trained model. Examples of model parameters
are the weights in an artificial neural network, the support vectors in a
support vector machine or the coefficients/weights in a linear or logistic 
regression.

**Hyperparameters** in turn, are 'configuration(s)' of a learning algorithm,
which cannot be estimated from data, but nevertheless need to be specified to 
help estimate the model parameters. The best value for a hyperparameter on a 
given problem is usually not known and therefore has to be either set manually, 
based on experience from a previous similar problem or set by using a 
heuristic (rule of thumb) or by being 'tuned'. Examples are the learning rate 
for training a neural network, the C and sigma hyperparameters for support 
vector machines or the number of estimators in a random forest. 

Manually specifying hyperparameters with Julearn is as simple as using the 
``PipelineCreator`` and add or change hyperparameters for each step in the 
pipeline. 

.. code-block:: python

  creator = (PipelineCreator()
           .add("zscore", with_mean=[True])
           .add("pca", n_components=[.2])
           .add("svm", problem_type="classification")
          )

  run_cross_validation(
    X=X, y=y, data=df, 
    model = creator
  )

<Add more hyperparameters, e.g. C for the SVM?>

Usable transformers or estimators can be seen under 
:ref:`available_pipeline_steps`. The basis for most of these steps are the
respective scikit-learn estimators or transformers. To see the valid 
hyperparameters for a certain transformer or estimator, just follow the 
respective link in :ref:`available_pipeline_steps` which will lead you to the 
scikit-learn documentation where you can read more about the hyperparameters.

.. _apply_to_feature_types:

Applying preprocessing only to certain feature types
----------------------------------------------------
Under :ref:`pipeline_creator` you might have wondered, how the 
``PipelineCreator`` makes things easier. Beside the very straight forward 
definition of hyperparameters, the ``PipelineCreator`` also helps to apply 
certain steps of the pipeline only to pre-defined types of data (see 
:ref:`data_usage` on how to pre-define types of data). This can for example be 
useful when one wants to apply a preprocessing steps only to 
a certain type of feature, like for example continous features. We here 
exemplarily apply a 'pca' only to the 'petal' features of the 'iris' dataset.

First, one needs to define the ``X_types`` to which the ``pca`` should be 
applied:

.. code-block:: python

    X_types = dict(petal=[ 'petal_length', 'petal_width']) 

Next, in the ``PipelineCreator`` one specifies at the respective step of the 
pipeline (in our case the ``pca``) that it should only be applied to these 
``X_types``:

.. code-block:: python

    creator = (PipelineCreator()
           .add("pca", apply_to="petal",  n_components=1)
           .add("svm", problem_type="classification")
          )

Finally, we again pass the defined ``X_types`` and the creator to 
:func:`.run_cross_validation`:

.. code-block:: python

    run_cross_validation(
      X=X, y=y, data=df, 
      X_types=X_types,
      model = creator
    )

In a slightly more complex use-case one might want to ``z-score`` all
features and apply a ``pca`` only to the 'petal' features. The 
``apply-to``-parameter can also receive a list of ``X_types``. To demonstrate 
this we split the petal features in two different types: 'xtype1' and 'xtype2'.
This also shows that ``X_types`` is a dictionary. Therefore, one can specify as +
many different ``X_types`` as wished in a key-value manner. The key is the name 
of the type and the list of column-names in the features that should belong to 
this type. Splitting the 'petal' columns in two different ``X_types`` is not 
necessary and only for demonstration purposes. We here only want to 
use the first component of the ``pca`` and therfore specify its ``n_components`` 
hyperparameter respectively.

.. code-block:: python

    X_types = dict(
      xtype1=[ 'petal_length'],
      xtype2=['petal_width']
      )

    creator = (PipelineCreator()
           .add("zscore", apply_to="*")
           .add("pca", apply_to=["xtype1", "xtype2"],  n_components=1)
           .add("svm", problem_type="classification")
          )

    run_cross_validation(
      X=X, y=y, data=df, 
      X_types = X_types,
      model = creator
    )

We covered how to set up a basic pipelines, how to use the ``PipelineCreator`` 
and how to use the 'apply_to'-parameter of the ``PipelineCreator`` and covered 
basics of hyperparameters. In the next step we will understand the returns of 
:func:`.run_cross_validation`, i.e. the model output and the scores of the 
performed corss-validation.


.. More information
.. ^^^^^^^^^^^^^^^^^

.. As mentioned above julearn allows the user to specify to which variable/columns
.. or variable/column types each transformer will be applied. To do so you
.. can adjust the ``apply_to`` hyperparameter which is added to all transformers
.. used in ``preprocess_X``. You can find such an example at #TODO
.. and find more information on hyperparameter tuning in
.. :doc:`hyperparameters <hyperparameters>` .


.. topic:: References:

	.. [#1] Boser, B. E., Guyon, I. M., & Vapnik, V. N., `"A training algorithm for optimal margin classifiers" <https://dl.acm.org/doi/pdf/10.1145/130385.130401>`_, COLT '92 Proceedings of the fifth annual workshop on Computational learning theory. 1992 Jul; 144-152.