"""
Grouped CV
=====================

This example uses the 'iris' dataset and performs GroupKFold
Cross-Validation for classification using Random Forest Classifier.

References
----------
Waskom, M.L., Frank, M.C., Wagner, A.D. (2016). Adaptive engagement of
cognitive control in context-dependent decision-making. Cerebral Cortex.


.. include:: ../../links.inc
"""
# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Shammi More <s.more@fz-juelich.de>
#          Kimia Nazarzadeh <k.nazarzadeh@fz-juelich.de>


#
# License: AGPL
# Importing the necessary Python libraries
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from seaborn import load_dataset
from sklearn.model_selection import GroupKFold

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from julearn.utils import configure_logging
from julearn import run_cross_validation
from julearn.model_selection import StratifiedGroupsKFold


###############################################################################
# Set the logging level to info to see extra information
configure_logging(level="INFO")

###############################################################################
# load the iris data from seaborn
df_iris = load_dataset('iris')

###############################################################################
# As features, we will use the sepal length, width and petal length.
# We will try to predict the species.
X = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
y = "species"

###############################################################################
# Define number of groups for CV and create group for stratification
number_groups = 4
groups = np.floor(np.linspace(0, number_groups, len(df_iris)))

###############################################################################
# Train classification model with stratification on data
cv_stratified = StratifiedGroupsKFold(n_splits=2)
scores, model = run_cross_validation(
    X=X, y=y, data=df_iris, groups=groups,
    model='rf', cv=cv_stratified, return_estimator="final")

###############################################################################
# Train classification model withot stratification on data
cv = GroupKFold(
    n_splits=2).split(
        df_iris, groups=groups)
scores, model = run_cross_validation(
    X=X, y=y, data=df_iris,
    model='rf', cv=cv, return_estimator="final")

###############################################################################
# Now we can compare the test score for model trained with and without
# stratification. We can combine the two outputs as pandas dataframes
scores_strat['model'] = 'With stratification'
scores['model'] = 'Without stratification'
df_scores = scores_strat[['test_score', 'model']]
df_scores  = pd.concat([df_scores, scores[['test_score', 'model']]])
