"""
Inspecting Random Forest models
===============================

This example uses the ``iris`` dataset, performs simple binary classification
using a Random Forest classifier and analyse the model.

.. include:: ../../links.inc
"""
# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
# License: AGPL

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from seaborn import load_dataset

from julearn import run_cross_validation
from julearn.utils import configure_logging

###############################################################################
# Set the logging level to info to see extra information.
configure_logging(level="INFO")

###############################################################################
# Random Forest variable importance
# ---------------------------------

df_iris = load_dataset("iris")

###############################################################################
# The dataset has three kind of species. We will keep two to perform a binary
# classification.
df_iris = df_iris[df_iris["species"].isin(["versicolor", "virginica"])]

X = ["sepal_length", "sepal_width", "petal_length"]
y = "species"


###############################################################################
# We will use a Random Forest classifier. By setting
# `return_estimator='final'`, the :func:`.run_cross_validation` function
# returns the estimator fitted with all the data.

scores, model_iris = run_cross_validation(
    X=X,
    y=y,
    data=df_iris,
    model="rf",
    preprocess="zscore",
    problem_type="classification",
    return_estimator="final",
)

###############################################################################
# This type of classifier has an internal variable that can inform us on how
# *important* is each of the features. Caution: read the proper ``scikit-learn``
# documentation :class:`~sklearn.ensemble.RandomForestClassifier` to understand
# how this learning algorithm works.
rf = model_iris["rf"]

to_plot = pd.DataFrame(
    {
        "variable": [x.replace("_", " ") for x in X],
        "importance": rf.feature_importances_,
    }
)

fig, ax = plt.subplots(1, 1, figsize=(6, 4))
sns.barplot(x="importance", y="variable", data=to_plot, ax=ax)
ax.set_title("Variable Importances for Random Forest Classifier")
fig.tight_layout()

###############################################################################
# However, some reviewers (including us), might wander about the
# variability of the importance of these features. In the previous example
# all the feature importances were obtained by fitting on the entire dataset,
# while the performance was estimated using cross validation.
#
# By specifying `return_estimator='cv'`, we can get, for each fold, the fitted
# estimator.

scores = run_cross_validation(
    X=X,
    y=y,
    data=df_iris,
    model="rf",
    preprocess="zscore",
    problem_type="classification",
    return_estimator="cv",
)

###############################################################################
# Now we can obtain the feature importance for each estimator (CV fold).
to_plot = []
for i_fold, estimator in enumerate(scores["estimator"]):
    this_importances = pd.DataFrame(
        {
            "variable": [x.replace("_", " ") for x in X],
            "importance": estimator["rf"].feature_importances_,
            "fold": i_fold,
        }
    )
    to_plot.append(this_importances)

to_plot = pd.concat(to_plot)

###############################################################################
# Finally, we can plot the variable importances for each fold.

fig, ax = plt.subplots(1, 1, figsize=(6, 4))
sns.swarmplot(x="importance", y="variable", data=to_plot, ax=ax)
ax.set_title(
    "Distribution of variable Importances for Random Forest "
    "Classifier across folds"
)
fig.tight_layout()
