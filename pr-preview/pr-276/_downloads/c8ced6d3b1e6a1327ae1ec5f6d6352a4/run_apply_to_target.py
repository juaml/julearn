"""
Transforming target variable with z-score
=========================================

This example uses the sklearn ``diabetes`` regression dataset, and transforms the
target variable, in this case, using z-score. Then, we perform a regression
analysis using Ridge Regression model.

"""
# Authors: Lya K. Paas Oliveros <l.paas.oliveros@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
#
# License: AGPL

import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

from julearn import run_cross_validation
from julearn.utils import configure_logging

from julearn.pipeline import PipelineCreator, TargetPipelineCreator

###############################################################################
# Set the logging level to info to see extra information.
configure_logging(level="INFO")

###############################################################################
# Load the diabetes dataset from ``sklearn`` as a ``pandas.DataFrame``.
features, target = load_diabetes(return_X_y=True, as_frame=True)

###############################################################################
# Dataset contains ten variables age, sex, body mass index, average  blood
# pressure, and six blood serum measurements (s1-s6) diabetes patients and
# a quantitative measure of disease progression one year after baseline which
# is the target we are interested in predicting.
print("Features: \n", features.head())
print("Target: \n", target.describe())

###############################################################################
# Let's combine features and target together in one dataframe and define X
# and y.
data_diabetes = pd.concat([features, target], axis=1)

X = ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]
y = "target"

###############################################################################
# Split the dataset into train and test.
train_diabetes, test_diabetes = train_test_split(data_diabetes, test_size=0.3)

###############################################################################
# Let's create the model. Since we will be transforming the target variable
# we will first need to create a TargetPipelineCreator for this.

target_creator = TargetPipelineCreator()
target_creator.add("zscore")

##############################################################################
# Now we can create the pipeline using a PipelineCreator.
creator = PipelineCreator(problem_type="regression")
creator.add(target_creator, apply_to="target")
creator.add("ridge")

scores, model = run_cross_validation(
    X=X,
    y=y,
    data=train_diabetes,
    model=creator,
    return_estimator="final",
    scoring="neg_mean_absolute_error",
)

print(scores.head(5))

###############################################################################
# Mean value of mean absolute error across CV
print(scores["test_score"].mean() * -1)
