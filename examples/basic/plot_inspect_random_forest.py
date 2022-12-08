""".

Inspecting Random Forest models
===============================

This example uses the 'iris' dataset, performs simple binary classification
using a Random Forest classifier. We will also analyse the features used
by the model to perform the classification.

References
----------
Fisher, R. A. (1936). The use of multiple measurements in taxonomic problems.
Annals of eugenics, 7(2), 179-188.

.. include:: ../../links.inc
"""
# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Nieto Nicolás <n.nieto@fz-juelich.de>
# License: AGPL
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from seaborn import load_dataset

from julearn import run_cross_validation
from julearn.utils import configure_logging
from sklearn.model_selection import StratifiedKFold

###############################################################################
# Set the logging level to info to see extra information
configure_logging(level='INFO')

###############################################################################
# Iris classification with Random Forest - variable importance analysis.
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#

###############################################################################
# Load iris dataset from `seaborn` library. It contains information about
# three flowers: `species`, `versicolor`,`virginica` and `setosa`.
# For each observation, it has four features: `sepal_length`, `sepal_width`,
# `petal_length` and `petal_width`.
#
# The data needs to be a `pandas.DataFrame`
df_iris = load_dataset('iris')

###############################################################################
# The dataset has three kind of species. We will keep only two of them
# to perform a binary classification.
df_iris = df_iris[df_iris['species'].isin(['versicolor', 'virginica'])]

###############################################################################
# As features, we will only use sepal length, width and petal length.
# We set a list with the name of the columns we want to use as features.
# Those are columns names of `df_iris`
X = ['sepal_length', 'sepal_width', 'petal_length']

###############################################################################

# As target, we will use the species, as we want to predict the specie with the
# information about the petal and sepal.
# We set a variable with the name of the column we want to use as a target.
# The target is also included in `df_iris`
y = 'species'

###############################################################################
# If we take a look at the data, we will find that the samples are sorted by
# the target. In that case, a problem will occur if we use a simple Kfold
# approach, as some splits will contain only samples from one class.
#
# To solve this, we will use a Stratified KFold approach, that takes care of
# this, and makes sure that every split have the same proportion of data for
# each class.
cv = StratifiedKFold(n_splits=5)

###############################################################################
# We will use a Random Forest classifier. We will set
# `return_estimator='final'`, so the :func:`.run_cross_validation` function
# will return the model fitted with all the data. We will then analyse the
# most relevant features used by the model.

scores, model_iris = run_cross_validation(
    X=X, y=y, data=df_iris, model='rf', preprocess='zscore', cv=cv,
    return_estimator='final')

###############################################################################
# This type of classifier has an internal variable that can inform us on how
# _important_ is each of the features. Please, read the proper scikit-learn
# documentation (`Random Forest`_) for more information.
rf = model_iris['rf']

to_plot = pd.DataFrame({
    'variable': [x.replace('_', ' ') for x in X],
    'importance': rf.feature_importances_
})

fig, ax = plt.subplots(1, 1, figsize=(6, 4))
sns.barplot(x='importance', y='variable', data=to_plot, ax=ax)
ax.set_title('Variable Importances for Random Forest Classifier')
fig.tight_layout()

###############################################################################
# However, one might wander about the variability of the importance of these
# features. In the previous example all the feature importances were obtained
# by fitting on the entire dataset, while the performance was estimated using
# cross validation. This set up is a better choice if what we want is a final
# model that is trained with all the available data and will be used over an
# unseen data.
#
# By specifying `return_estimator='cv'`, we can get, for each fold, the fitted
# estimator and analyse how important were each feature for each estimator.
# If the feature importance large vary across folds, this could indicate a
# typical case of overffiting, where the estimator heavily relies in one
# feature.
#
# Note that now, all the models will be saved in a 'estimator' column.

scores = run_cross_validation(
    X=X, y=y, data=df_iris, model='rf',  preprocess='zscore', cv=cv,
    return_estimator='cv')

###############################################################################
# Now we can obtain the feature importance for each estimator, trained over
# different CV folds and see the variance of the feature impotance across each
# fold and repetition. We will obtain a total of 5 points, corresponding to
# each fold.

to_plot = []
for i_fold, estimator in enumerate(scores['estimator']):
    this_importances = pd.DataFrame({
        'variable': [x.replace('_', ' ') for x in X],
        'importance': estimator['rf'].feature_importances_,
        'fold': i_fold
    })
    to_plot.append(this_importances)

to_plot = pd.concat(to_plot)

###############################################################################
# Finally, we can plot the variable importances for each fold and repetition.

fig, ax = plt.subplots(1, 1, figsize=(6, 4))
sns.swarmplot(x='importance', y='variable', data=to_plot, ax=ax)
ax.set_title('Distribution of variable Importances for Random Forest '
             'Classifier across folds')
fig.tight_layout()
