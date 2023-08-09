# Authors: Vera Komeyer <v.komeyer@fz-juelich.de>
#          Fede Raimondo <f.raimondo@fz-juelich.de>
# License: AGPL

"""
Applying preprocessing to the target
------------------------------------

What we covered so far is how to apply preprocessing to the features and train
a model in a cv-conistent manner by building a pipeline.
However, sometimes one wants to apply preprocessing to the target. For example,
when having a regression-task (continuous target variable), one might want to
predict the z-scored target.
This can be achieved by using a :class:`.TargetPipelineCreator`
as a step in the general pipeline.

Let's start by loading the data and importing the required modules:
"""
import pandas as pd
from julearn import run_cross_validation
from julearn.pipeline import PipelineCreator, TargetPipelineCreator
from sklearn.datasets import load_diabetes

###############################################################################
# Load the diabetes dataset from ``scikit-learn`` as a ``pandas.DataFrame``
features, target = load_diabetes(return_X_y=True, as_frame=True)

print("Features: \n", features.head())
print("Target: \n", target.describe())

data_diabetes = pd.concat([features, target], axis=1)

X = ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]
y = "target"

X_types = {
    "continuous": ["age", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"],
    "categorical": ["sex"],
}

##############################################################################
# We first create a :class:`.TargetPipelineCreator`:

target_creator = TargetPipelineCreator()
target_creator.add("zscore")

print(target_creator)

##############################################################################
# Next, we create the general pipeline using a :class:`.PipelineCreator`. We
# pass the ``target_creator`` as a step of the pipeline and specify that it
# should only be applied to the ``target``, which makes it clear for ``julearn``
# to only apply it to ``y``:

creator = PipelineCreator(
    problem_type="regression", apply_to=["categorical", "continuous"]
)
creator.add(target_creator, apply_to="target")
creator.add("svm")
print(creator)

###############################################################################
# This ``creator`` can then be passed to :func:`.run_cross_validation`:

scores = run_cross_validation(
    X=X, y=y, data=data_diabetes, X_types=X_types, model=creator
)

print(scores)

###############################################################################
# All transformers in (:ref:`available_transformers`) can be used for both,
# feature and target transformations. However, features transformations can be
# directly specified as step in the :class:`.PipelineCreator`, while target
# transformations have to be specified using the
# :class:`.TargetPipelineCreator`, which is then passed to the overall
# :class:`.PipelineCreator` as an extra step.
