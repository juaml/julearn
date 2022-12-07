"""
Inspecting SVM models
=====================

This example uses the 'fmri' dataset, performs simple binary classification
using a Support Vector Machine classifier and analyse the model.


References
----------
Waskom, M.L., Frank, M.C., Wagner, A.D. (2016). Adaptive engagement of
cognitive control in context-dependent decision-making. Cerebral Cortex.


.. include:: ../../links.inc
"""
# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Nieto Nicolás <n.nieto@fz-juelich.de>
# License: AGPL

# %%
import numpy as np

from sklearn.model_selection import GroupShuffleSplit

import matplotlib.pyplot as plt
import seaborn as sns
from seaborn import load_dataset

from julearn import run_cross_validation
from julearn.utils import configure_logging

###############################################################################
# Set the logging level to info to see extra information
configure_logging(level='INFO')

###############################################################################
# Dealing with Cross-Validation techniques
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# We will use a seaborn dataset that contains fmri data.
df_fmri = load_dataset('fmri')

###############################################################################
# First, lets get some information on what the dataset has:
#
print(df_fmri.head())

###############################################################################
# From this information, we can infer that it is an fMRI study in which there
# were several subjects, timepoints, events and signal extracted from several
# brain regions.
#
# Lets check how many kinds of each we have.

print('Events names')
print(df_fmri['event'].unique())
print('Available regions')
print(df_fmri['region'].unique())
print('Number of timepoints')
print(sorted(df_fmri['timepoint'].unique()))
print('Available participants')
print(df_fmri['subject'].unique())

###############################################################################
# We have data from parietal and frontal regions during 2 types of events
# (*cue* and *stim*) during 18 timepoints and for 14 subjects.
# Lets see how many samples we have for each condition

print(df_fmri.groupby(['subject', 'timepoint', 'event', 'region']).count())
print("Number of samples for each condition: %d" % np.unique(df_fmri.groupby(
    ['subject', 'timepoint', 'event', 'region']).count().values))

###############################################################################
# We have exactly one value per condition.
#
# We will build a model that, given both parietal and frontal signal,
# predicts if the event was a *cue* or a *stim*.
#
# First we define our X and y variables. X must be a list of strings with the
# column names of the features used as input. y must be a string with the
# target column name.
X = ['parietal', 'frontal']
y = 'event'

###############################################################################
# We need to *pivot* the table as both *parietal* and *frontal* must be columns
#
# The values of *region* will be the columns. The column *signal* will be the
# values. And the columns *subject*, *timepoint* and *event* will be the index

df_fmri = df_fmri.pivot(
    index=['subject', 'timepoint', 'event'],
    columns='region',
    values='signal')

df_fmri = df_fmri.reset_index()

###############################################################################
# We will see how the data is organized after the *pivot*.

print(df_fmri.head())

###############################################################################
# Now, we will built a classification pipeline. We will preprocess the data
# using a *zcore* and use Support Vector Machine as a classification model.

scores = run_cross_validation(X=X, y=y, preprocess_X='zscore', data=df_fmri,
                              model='svm')

print('Mean Test Accuracy: %f' % scores['test_score'].mean())

###############################################################################
# This results indicate that we can decode the kind of event by looking at
# the *parietal* and *frontal* signal. However, in the current set up,
# that claim is true only if we have some data from the same subject
# already acquired, but we can not make a claim over an unseen participant.
#
# The problem with the current set up is that we split the data randomly
# into 5 folds (default, see :func:`.run_cross_validation`). This means that
# data from one subject could be both in the training and the testing set.
# If this is the case, then the model can learn the subjects' specific
# characteristics and apply it to the testing set.
# Thus, it is not true that we can decode it for an unseen subject, but for an
# unseen timepoint for a subject that for whom we already have data.
#
# To test for unseen subject, we need to make sure that all the data from each
# subject is either on the training or the testing set, but not in both.
#
# For doing so we will use the scikit-learn's GroupShuffleSplit
# (see `Cross Validation`_).
# and specify which is the grouping column using the `group` parameter.
# This will make that the data coming from one subject (group) are keeped only
# in the train split *or* the test split.
#
# We will set `return_estimator='final'` in the :func:`.run_cross_validation`s
# function, so it will return the final estimator fitted with *all* the data.
# We will use this # analysis the final model.
cv = GroupShuffleSplit(n_splits=5, test_size=0.5, random_state=42)

scores, model = run_cross_validation(
    X=X, y=y, data=df_fmri, model='svm', preprocess_X='zscore', cv=cv,
    groups='subject', return_estimator='final')

print('Mean Test Accuracy with gruped CV: %f' % scores['test_score'].mean())

###############################################################################
# After testing on unseen subjects, we can now claim that given a new subject,
# we can predict the kind of event from the `parietal` and `frontal`
# information
#
# Lets do a visualization on how the features interact with each other
# and what the preprocessing part of our model is doing.
#
# As we aks the function :func:`.run_cross_validation`) to return the final
# model, the 'preprocess' step is stored in our object 'model'

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
sns.scatterplot(x='parietal', y='frontal', hue='event', data=df_fmri,
                ax=axes[0], s=5)
axes[0].set_title('Raw data')

pre_X, pre_y = model.preprocess(df_fmri[X], df_fmri[y])
pre_df = pre_X.join(pre_y)
sns.scatterplot(x='parietal', y='frontal', hue='event', data=pre_df,
                ax=axes[1], s=5)
axes[1].set_title('Preprocessed data')

###############################################################################
# In this case, the preprocessing is nothing more than a `Standard Scaler`_.
#
# It seems that the data is not quite linearly separable. Lets now visualize
# how the SVM does this complex task.
clf = model['svm']
ax = sns.scatterplot(x='parietal', y='frontal', hue='event', data=pre_df, s=5)

xlim = ax.get_xlim()
ylim = ax.get_ylim()

# create grid to evaluate model
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)
a = ax.contour(XX, YY, Z, colors='k', levels=[0], alpha=0.5, linestyles=['-'])
ax.set_title('Preprocessed data with SVM decision function boundaries')
plt.show()
# %%
