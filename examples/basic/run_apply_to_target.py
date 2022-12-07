"""
Transforming target variable with z-score
==========================

This example uses the sklearn "diabetes" regression dataset,
and transforms the target variable, in this case, using z-score.
Then, we perform a regression analysis using Ridge Regression model.

"""
# Authors: Lya K. Paas Oliveros <l.paas.oliveros@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
#
# License: AGPL

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from julearn import run_cross_validation
from julearn.utils import configure_logging
from julearn.pipeline import PipelineCreator # this is crucial for creating the model in the new version
from julearn.inspect import preprocess

###############################################################################
# Set the logging level to info to see extra information
configure_logging(level='INFO')

###############################################################################
# Load the diabetes dataset from sklearn as a pandas dataframe
features, target = load_diabetes(return_X_y=True, as_frame=True)

###############################################################################
# Dataset contains ten variables age, sex, body mass index, average  blood
# pressure, and six blood serum measurements (s1-s6) diabetes patients and
# a quantitative measure of disease progression one year after baseline which
# is the target we are interested in predicting.

print('Features: \n', features.head())
print('Target: \n', target.describe())

###############################################################################
# Let's combine features and target together in one dataframe and define X
# and y
data_diabetes = pd.concat([features, target], axis=1)

X = ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
y = 'target'

###############################################################################
# Calculate correlations between the features/variables and plot it as heat map
corr = data_diabetes.corr()
fig, ax = plt.subplots(1, 1, figsize=(10, 7))
sns.set(font_scale=1.2)
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns,
            annot=True, fmt="0.1f")


###############################################################################
# Split the dataset into train and test
train_diabetes, test_diabetes = train_test_split(data_diabetes, test_size=0.3)


###############################################################################
# Let's create the model. To note, in PipelineCreator the only reserved key is target.
# That means, there is no need to name it, i.e., by stating apply_to="target",
# it already knows which is the target.
# Here, it is important that if you define the PipelineCreator you include the model and do not define the model in run_cross_validation
creator = (PipelineCreator()
           .add("zscore", apply_to="target")
           .add("ridge", problem_type="regression")
          )

scores, model = run_cross_validation(
            X=X,
            y=y,
            data=train_diabetes,
            model=creator,
            return_estimator="final",
            scoring='neg_mean_absolute_error'
)

print(scores.head(5))

###############################################################################
# Mean value of mean absolute error across CV
print(scores['test_score'].mean() * -1)