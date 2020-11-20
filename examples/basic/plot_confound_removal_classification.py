"""
Confound Removal
============================

This example uses the 'iris' dataset, performs simple binary classification
with and without confound removal using a Random Forest classifier.

"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn import load_dataset
from sklearn.model_selection import StratifiedKFold

from julearn import run_cross_validation
from julearn.utils import configure_logging

###############################################################################
# Set the logging level to info to see extra information
configure_logging(level='INFO')

###############################################################################
# load the iris data from seaborn
df_iris = load_dataset('iris')

###############################################################################
# The dataset has three kind of species. We will keep two to perform a binary
# classification.
df_iris = df_iris[df_iris['species'].isin(['versicolor', 'virginica'])]


###############################################################################
# As features, we will use the sepal length, width and petal length and use
# petal width as confound.

X = ['sepal_length', 'sepal_width', 'petal_length']
y = 'species'
confound = 'petal_width'

###############################################################################
# Use stratified 10 fold CV
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=200)

###############################################################################
# First, we will train a model without performing confound removal on features
# Note: confounds=None by default
scores_ncr = run_cross_validation(X=X, y=y, data=df_iris, model='rf', cv=cv,
                                  scoring='accuracy', return_estimator='cv',
                                  seed=200)

###############################################################################
# Accuracy for each fold
print(scores_ncr['test_accuracy'])

###############################################################################
# Next, we train a model after performing confound removal on the features
# Note: we use the same random seed as before to create same CV splits
scores_cr = run_cross_validation(X=X, y=y, confounds=confound, data=df_iris,
                                 model='rf', preprocess_X='remove_confound',
                                 cv=cv, scoring='accuracy',
                                 return_estimator='cv',
                                 seed=200)

###############################################################################
# Accuracy for each fold
print(scores_cr['test_accuracy'])

###############################################################################
# Let's compare CV accuracy for each fold across the two models
# NCR: no confound removal
# CVCR: Cross-validated confound removal

scores_df = pd.DataFrame({'NCR': scores_ncr['test_accuracy'],
                          'CVCR': scores_cr['test_accuracy']})
sns.set(style="darkgrid", font_scale=1.2)
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
sns.boxplot(data=scores_df, width=0.3, palette="colorblind")
sns.stripplot(data=scores_df, jitter=True, dodge=True, marker='o', alpha=0.6,
              color='k')
ax.set(ylabel='Accuracy', title='Model performance without and with confound '
                                'removal')
plt.savefig('1.png')


###############################################################################
# Let's compare the importance of each feature in classification with and
# without condound removal

###############################################################################
# Get feature importance for each CV fold for model trained without confound
# removal (NCR)

ncr_to_plot = []
for i_fold, estimator in enumerate(scores_ncr['estimator']):
    this_importances = pd.DataFrame({
        'features': [x.replace('_', ' ') for x in X],
        'NCR': estimator['rf'].feature_importances_,
        'fold': i_fold
    })
    ncr_to_plot.append(this_importances)
ncr_to_plot = pd.concat(ncr_to_plot)

###############################################################################
# Get feature importance for each CV fold for model trained with confound
# removal on features (CVCR)

cr_to_plot = []
for i_fold, estimator in enumerate(scores_cr['estimator']):
    this_importances = pd.DataFrame({
        'features': [x.replace('_', ' ') for x in X],
        'CVCR': estimator['rf'].feature_importances_,
        'fold': i_fold
    })
    cr_to_plot.append(this_importances)
cr_to_plot = pd.concat(cr_to_plot)

###############################################################################
# Combine both the dataframes (ncr_to_plot and cr_to_plot) for further plotting

df = pd.merge(left=ncr_to_plot, right=cr_to_plot, how='inner')
sns.set(style="darkgrid", font_scale=1.2)
fig, ax1 = plt.subplots(figsize=(15, 10))
tidy = df.melt(id_vars=['features', 'fold'])
sns.swarmplot(x='value', y='features', hue='variable', data=tidy, ax=ax1,
              size=8)
ax1.set(xlabel='Importance', title='Feature importance without and with '
                                   'confound removal')
plt.savefig('2.png')
