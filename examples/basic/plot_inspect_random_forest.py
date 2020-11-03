"""
Inspecting Random Forest models
===============================

This example uses the 'iris' dataset, performs simple binary classification
using a Random Forest classifier and analyse the model.

.. include:: ../links.inc
"""
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from seaborn import load_dataset

from julearn import run_cross_validation

###############################################################################
# Random Forest variable importance
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#

df_iris = load_dataset('iris')

###############################################################################
# The dataset has three kind of species. We will keep two to perform a binary
# classification.
df_iris = df_iris[df_iris['species'].isin(['versicolor', 'virginica'])]

X = ['sepal_length', 'sepal_width', 'petal_length']
y = 'species'


###############################################################################
# We will use a Random Forest classifier. By setting `return_estimator=True`,
# the :func:`.run_cross_validation` function return the estimator fitted with
# all the data.

scores, model_iris = run_cross_validation(X=X, y=y, data=df_iris, model='rf',
                                          return_estimator=True)

###############################################################################
# This type of classifier has an internal variable that can inform us on how
# _important_ is each of the features. Caution: read the proper scikit-learn
# documentation (`Random Forest`_)
rf = model_iris['rf']

to_plot = pd.DataFrame({
    'variable': [x.replace('_', ' ') for x in X],
    'importance': rf.feature_importances_
})

fig, ax = plt.subplots(1, 1, figsize=(6, 4))
sns.barplot(x='importance', y='variable', data=to_plot, ax=ax)
ax.set_title('Variable Importances for Random Forest Classifier')
fig.tight_layout()
