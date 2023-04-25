.. include:: ../links.inc

.. _pipeline_usage:

Pipeline
========

So far we know how to pass input data to :func:`.run_cross_validation` 
(see :ref:`data_usage`). Now, we will have a look at the implementation of 
basic machine learning pipelines.

A machine learning pipeline is a process to automate the workflow of 
building a predictive model. It can be thought of as a combination of pipes and 
filters. At a pipeline's starting point, the raw data is fed into this
pipeline. Different filters inside the pipeline modify the data. The output of
a machine learning pipeline is a model that can be used 
for predictions on unseen data. Before using the pipeline
to predict new data, the pipeline has to be trained or fitted on data. We call
this, as scikit-learn does, _fitting_ the pipeline.

Importantly, since we want to test how well our model _predicts_ new data, we
cannot predict using the same data we used for _fitting_. Thus, we need to have
separate data for fitting and predicting. This is where cross validation comes
in handy. Cross validation is a technique to split the data into a training and
validation data set. The training data set is used to fit the pipeline, while the
validation data set is used to predict the data. The predictions are then compared
to the true values of the validation data set, obtaining an estimation of the
prediction performance of the model (see also :ref:`why_cv`).

Julearn aims to provide a user-friendly way to build and evaluate complex
machine learning pipelines. The :func:`.run_cross_validation` function is the
entry point to safely evaluate pipelines by making it easy to specify, 
costumize and train the pipeline. We first have a look at the most
basic pipeline, only consisting of a machine learning algorithm. Then we will
make the pipeline incrementally more complex.

.. _basic_cv:

Basic cross validation with Julearn
-----------------------------------

One important aspect when building machine learning models is the selection of
a learning algorithm. This can be specified in :func:`.run_cross_validation`
by setting the ``model`` parameter. This parameter can be any scikit-learn
compatible learning algorithm. However, Julearn provides a built-in list of
:ref:`available_models` that can be specified by name (see _Name (str)_ 
column in :ref:`available_models`). For example, we 
can simply set ``model=="svm"`` to use a support vector machine (SVM) [#1]_. 

.. code-block:: python

    run_cross_validation(
        X=X,
        y=y,
        data=df,
        model="svm",
        problem_type="classification",
    )

You will notice that this code indicates an extra parameter ``problem_type``.
This is because in machine learning, one can distinguish between regression
problems -when predicting a continous outcome- and classification problems 
-for discrete class label predictions-. Therefore, 
:func:`.run_cross_validation` additionally needs to know which problem type we
are interested in. This is done by specifying the parameter ``problem_type``. 
The possible values are ``classification`` and ``regression`` and importantly,
there is no default specified, so you have to explicitely set it. In the
example we are interested in predicting the species
(see ``y`` in :ref:`data_usage`), i.e. a discrete class label. Therefore, 
the ``problem_type`` is ``classification``. 

Et voilà, your first machine learning pipeline is ready to go.

(Feature) preprocessing
-----------------------
There are cases in which the input data, and in particular the features, 
should be transformed before passing them to the learning algorithm. One 
scenario can be that certain learning algorithms need the features in a 
specific form, for example in standardized form, so that the data resemble a 
normal distribution. This can be achieved by first z-scoring (or standard 
scaling) the features (see :ref:`available_scalers`). 

Importantly, in a machine learning workflow all transformations done to the 
data have to be done in a cv-consistent way. That means, that one should do 
steps like feature preprocessing on the training data of each respective cross 
validation fold and then only apply the parameters of the transformation to the 
validation data of the respective fold. One should **never** do preprocessing 
on the entire dataset and then do cross validation on the already preprocessed 
features (or more
generally transformed data) because this leads to leakage of information from
the validation data into the model. This is exactly where
:func:`.run_cross_validation` comes in handy, because you can simply add your
wished preprocessing step (:ref:`available_transformers`) and it 
takes care of doing the respective transformations in a cv-consistent manner.   

.. code-block:: python

  run_cross_validation(
      X=X,
      y=y,
      data=df, 
      preprocess="zscore",
      model="svm",
      problem_type="classification",
  )

.. note::
  Learning algorithms (what we specified in the `model` parameter), are
  estimators. Preprocessing steps however, are usually transformers, because 
  they transform the input data in a certain way. Therefore the parameter 
  description in the api of :func:`.run_cross_validation`, 
  defines valid input for the `preprocess` parameter as `TransformerLike`.

  preprocess : str, TransformerLike or list or PipelineCreator | None
          Transformer to apply to the features. If string, use one of the
          available transformers. If list, each element can be a string or
          scikit-learn compatible transformer. If None (default), no
          transformation is applied.

But what if we want to add more pre-processing steps? 
An examplary scenario can be, that there are many features available, so that one
wants to first reduce the dimensionality of the features before passing them
to the learning algorithm. A commonly used approach is a principal component
analysis (PCA, see :ref:`available_decompositions`). If we nevertheless 
want to keep our previously applied z-scoring, we can simply add the PCA as 
another preprocessing step as follows:

.. code-block:: python

  run_cross_validation(
      X=X,
      y=y,
      data=df, 
      preprocess=["zscore", "pca"],
      model="svm",
      problem_type="classification",
  )

This is nice, but with more steps added to the pipeline this can become 
intransparent. To simplify building complex pipelines, Julearn provides a 
:class:`.PipelineCreator` which helps keeping things neat.

.. _pipeline_creator:

Pipeline specification made easy with the :class:`.PipelineCreator`
-------------------------------------------------------------------

The :class:`.PipelineCreator` is a class that helps the user create complex
pipelines with straightforward usage by stepwise adding subsequent steps of 
the pipeline. 

Both the pre-processing steps and the passed learning-algorithm 
are subsequent steps of the pipeline. Following this logic, we can add 
both the specified preprocessing steps and the learning algorithm to the 
pipeline creator to define the pipeline.

.. note::
  The :class:`.PipelineCreator` always has to be initialized with the 
  ``problem_type`` parameter, which can be either ``classification`` or 
  ``regression``.

Let's re-write the previous example, using the :class:`.PipelineCreator`.

We start by creating an instance of the :class:`.PipelineCreator`, and
setting the ``problem_type`` parameter to ``classification``.

.. code-block:: python

  creator = PipelineCreator(problem_type="classification")

Then we use the ``add`` method and add every desired step to the pipeline. Both,
the pre-processing steps and the learning algorithm are added in the same way.
As above one can use the names of the step as indicated in 
:ref:`available_pipeline_steps`.

.. code-block:: python

    creator.add("zscore")
    creator.add("pca")
    creator.add("svm")
    
We then pass the ``creator`` to :func:`.run_cross_validation` as the 
``model`` parameter. We do not need to (and cannot) specify the ``preprocess``
parameter.

.. code-block:: python

    run_cross_validation(
        X=X,
        y=y,
        data=df, 
        model=creator
    )

.. note::
  If the argument passed to the ``model``-parameter is a PipelineCreator, 
  the preprocess parameter should be None. Otherwise, an error will be raised. 
  This can be done by just **not** specifying it as the default is None.


Awesome! We covered how to create a basic machine learning pipeline and 
even added multiple feature pre-preprocessing steps. 

Let's jump to the next important aspect in the process of building a machine
learning model: **Hyperparameters**. We here cover the basics of setting 
hyperparameters. If you want to know more about tuning (or optimizing) 
hyperparameters, please have a look at :ref:`hp_tuning`.

How to specify hyperparameters
------------------------------

If you are new to machine learning, the section heading might confuse you: 
Parameters, hyperparameters - aren't we doing machine learning, so shouldn't 
the model learn all our parameters? Well, yes and no. Yes, it should learn 
parameters. However, hyperparameters and parameters are two different things.

A **model parameter** is a variable that is internal to the learning
algorithm and we want to learn or estimate its value from the data, which in 
turn means that they are not set manually. They are required by the model and 
are often saved as part of the fitting process. Examples of model parameters
are the weights in an artificial neural network, the support vectors in a
support vector machine or the coefficients/weights in a linear or logistic 
regression.

**Hyperparameters** in turn, are _configuration(s)_ of a learning algorithm,
which cannot be estimated from data, but nevertheless need to be specified to 
help estimate the model parameters. The best value for a hyperparameter on a 
given problem is usually not known and therefore has to be either set manually, 
based on experience from a previous similar problem, set by using a 
heuristic (rule of thumb) or by being _tuned_. Examples are the learning rate 
for training a neural network, the ``C`` and ``sigma`` hyperparameters for
support vector machines or the number of estimators in a random forest. 

Manually specifying hyperparameters with Julearn is as simple as using the 
:class:`.PipelineCreator` and add or change hyperparameters for each step in the 
pipeline. 

.. code-block:: python

    creator = PipelineCreator(problem_type="classification")
    creator.add("zscore", with_mean=[True])
    creator.add("pca", n_components=[.2])
    creator.add("svm")

    run_cross_validation(
        X=X,
        y=y,
        data=df,
        model=creator,
    )

Usable transformers or estimators can be seen under 
:ref:`available_pipeline_steps`. The basis for most of these steps are the
respective scikit-learn estimators or transformers. To see the valid 
hyperparameters for a certain transformer or estimator, just follow the 
respective link in :ref:`available_pipeline_steps` which will lead you to the 
`scikit-learn`_ documentation where you can read more about the respective 
hyperparameters.

In many cases one wants to specify more than one hyperparameter. This can be 
done by passing each hyperparameter separated by a comma. For the ``svm`` we 
could for example specify the ``C`` and the kernel hyperparameter like this:

.. code-block:: python

  creator = (PipelineCreator(problem_type="classification")
           .add("zscore", with_mean=[True])
           .add("pca", n_components=[.2])
           .add("svm", C=[.9], kernel=['linear'])
          )

  run_cross_validation(
    X=X,
    y=y,
    data=df, 
    model=creator,
  )


.. _apply_to_feature_types:

Applying preprocessing only to certain feature types
----------------------------------------------------

Under :ref:`pipeline_creator` you might have wondered, how the 
:class:`.PipelineCreator` makes things easier. Beside the very straight forward 
definition of hyperparameters, the :class:`.PipelineCreator` also helps to apply 
certain steps of the pipeline only to pre-defined types of data (see 
:ref:`data_usage` on how to pre-define types of data). This can be useful,
for example, when one wants to apply a preprocessing step only to a certain
type of feature, like for example continous features. We here exemplarily 
apply a _PCA_ only to the _petal_ features of the _iris_ dataset.

First, one needs to define the ``X_types`` to which the ``pca`` should be 
applied:

.. code-block:: python

    X_types = {"petal": ["petal_length", "petal_width"]}

Next, in the :class:`.PipelineCreator`, we specify the ``apply_to`` parameter
at the respective step of the pipeline (in our case the ``pca``) that it should
only be applied to these ``X_types``:

.. code-block:: python

    creator = PipelineCreator(problem_type="classification")
    creator.add("pca", apply_to="petal", n_components=1)
    creator.add("svm")

We here additionally had specified as a hyperparameter of the _PCA_ 
that we want to use only the first component. For the ``svm`` we used 
the default hyperparameters.

Finally, we again pass the defined ``X_types`` and the ``creator`` to 
:func:`.run_cross_validation`:

.. code-block:: python

    run_cross_validation(
        X=X,
        y=y,
        data=df, 
        X_types=X_types,
        model=creator
    )

In a slightly more complex use-case, one might want to ``z-score`` all features
and apply a ``pca`` only to the _petal_ features. The ``apply-to`` parameter
can also receive a list of ``X_types``. To demonstrate this we split the
features in two different types: _petal_ and _sepal_. This also shows that
``X_types`` is a dictionary in which one can specify as many different
``X_types`` as wished in a key-value manner. The key is the name of the type 
and the value a list of column-names in the features that should belong to 
this type. Splitting the _petal_ columns in two different ``X_types`` is not 
necessary and only for demonstration purposes. We here only want to 
use the first component of the ``pca`` and therfore specify its ``n_components`` 
hyperparameter respectively.

.. code-block:: python

    X_types = {
        "petal": ["petal_length", "petal_width"],
        "sepal": ["sepal_length", "sepal_width"]
    }

    creator = PipelineCreator(problem_type="classification")
    creator.add("zscore", apply_to=["petal", "sepal"])
    creator.add("pca", apply_to="petal", n_components=1)
    creator.add("svm")

    run_cross_validation(
        X=X,
        y=y,
        data=df, 
        X_types=X_types,
        model=creator
    )

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

We covered how to set up basic pipelines, how to use the 
:class:`.PipelineCreator`, how to use the ``apply_to`` parameter of the 
:class:`.PipelineCreator` and covered basics of hyperparameters. Additional, 
we saw a basic use-case of target pre-processing. In the next
step we will understand the returns of :func:`.run_cross_validation`, i.e. the 
model output and the scores of the performed cross-validation.

.. topic:: References:

	.. [#1] Boser, B. E., Guyon, I. M., & Vapnik, V. N., `"A training algorithm for optimal margin classifiers" <https://dl.acm.org/doi/pdf/10.1145/130385.130401>`_, COLT '92 Proceedings of the fifth annual workshop on Computational learning theory. 1992 Jul; 144-152.