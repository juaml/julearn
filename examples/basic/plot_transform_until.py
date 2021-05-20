"""
Preprocessing with zscore and PCA
===================================

This example uses the 'iris' dataset, performs simple binary
classification after the features are normalized using zscore and feature
reduction using PCA. We will check the features after each preprocessing
step.

"""

# Authors: Shammi More <s.more@fz-juelich.de>
#
# License: AGPL

import matplotlib.pyplot as plt
import seaborn as sns
from seaborn import load_dataset

from julearn import run_cross_validation
from julearn.utils import configure_logging

###############################################################################
# Set the logging level to info to see extra information
configure_logging(level='INFO')

###############################################################################
# Load the iris data from seaborn
df_iris = load_dataset('iris')

###############################################################################
# The dataset has three kind of species. We will keep two to perform a binary
# classification.

df_iris = df_iris[df_iris['species'].isin(['versicolor', 'virginica'])]

###############################################################################
# We will use the sepal length, width and petal length and
# petal width as features and predict the species

X = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
y = 'species'

###############################################################################
# Let's look at the summary statistics of the raw features
print('Summary Statistics of the raw features : \n', df_iris.describe())

###############################################################################
# We will preprocess the features (zscore and PCA) first before we train a
# random forest model

# Define the model parameters and preprocessing steps first
model_params = {'pca__n_components': 2, 'rf__n_estimators': 200, 'cv': 5}
preprocess_X = ['zscore', 'pca']

scores, model = run_cross_validation(
    X=X, y=y, data=df_iris, model='rf', preprocess_X=preprocess_X,
    scoring=['accuracy', 'roc_auc'], model_params=model_params,
    return_estimator='final', seed=200)

###############################################################################
# We can now look at zscored features

zscored_X, zscored_y = model.preprocess(df_iris[X], df_iris[y], until='zscore')
zscored_df = zscored_X.join(zscored_y)

###############################################################################
# Let's look at the summary statistics of the zscored features. We see here
# that the mean of all the features is zero and standard deviation is one
print('Summary Statistics of the zscored features : \n', zscored_df.describe())

###############################################################################
# We can also look at the features pre-processed until pca. Notice that the
# column names have changed in this new dataframe

pre_X, pre_y = model.preprocess(df_iris[X], df_iris[y], until='pca')
print('Features after PCA : \n', pre_X)
pre_df = pre_X.join(pre_y)

###############################################################################
# Let's plot scatter plots for raw features and the PCA components

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
sns.scatterplot(x='sepal_length', y='sepal_width', data=df_iris, hue='species',
                ax=axes[0])
axes[0].set_title('Raw features')
sns.scatterplot(x='pca_component_0', y='pca_component_1', data=pre_df,
                hue='species', ax=axes[1])
axes[1].set_title('PCA components')
