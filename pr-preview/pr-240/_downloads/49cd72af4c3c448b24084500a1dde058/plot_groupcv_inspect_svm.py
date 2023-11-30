"""
Inspecting SVM models
=====================

This example uses the ``fmri`` dataset, performs simple binary classification
using a Support Vector Machine classifier and analyse the model.

References
----------

  Waskom, M.L., Frank, M.C., Wagner, A.D. (2016). Adaptive engagement of
  cognitive control in context-dependent decision-making. Cerebral Cortex.

.. include:: ../../links.inc
"""
# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Shammi More <s.more@fz-juelich.de>
# License: AGPL

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit

import matplotlib.pyplot as plt
import seaborn as sns
from seaborn import load_dataset

from julearn import run_cross_validation
from julearn.utils import configure_logging
from julearn.inspect import preprocess

###############################################################################
# Set the logging level to info to see extra information.
configure_logging(level="INFO")


###############################################################################
# Dealing with Cross-Validation techniques
# ----------------------------------------

df_fmri = load_dataset("fmri")

###############################################################################
# First, lets get some information on what the dataset has:

print(df_fmri.head())

###############################################################################
# From this information, we can infer that it is an fMRI study in which there
# were several subjects, timepoints, events and signal extracted from several
# brain regions.
#
# Lets check how many kinds of each we have.

print(df_fmri["event"].unique())
print(df_fmri["region"].unique())
print(sorted(df_fmri["timepoint"].unique()))
print(df_fmri["subject"].unique())

###############################################################################
# We have data from parietal and frontal regions during 2 types of events
# (*cue* and *stim*) during 18 timepoints and for 14 subjects.
# Lets see how many samples we have for each condition

print(df_fmri.groupby(["subject", "timepoint", "event", "region"]).count())
print(
    np.unique(
        df_fmri.groupby(["subject", "timepoint", "event", "region"])
        .count()
        .values
    )
)

###############################################################################
# We have exactly one value per condition.
#
# Lets try to build a model, that uses parietal and frontal signal to predicts
# whether the event was a *cue* or a *stim*.
#
# First we define our X and y variables.
X = ["parietal", "frontal"]
y = "event"

###############################################################################
# In order for this to work, both *parietal* and *frontal* must be columns.
# We need to *pivot* the table.
#
# The values of *region* will be the columns. The column *signal* will be the
# values. And the columns *subject*, *timepoint* and *event* will be the index
df_fmri = df_fmri.pivot(
    index=["subject", "timepoint", "event"], columns="region", values="signal"
)

df_fmri = df_fmri.reset_index()

###############################################################################
# Here we want to zscore all the features and then train a Support Vector
# Machine classifier.

scores = run_cross_validation(
    X=X,
    y=y,
    data=df_fmri,
    preprocess="zscore",
    model="svm",
    problem_type="classification",
)

print(scores["test_score"].mean())

###############################################################################
# This results indicate that we can decode the kind of event by looking at
# the *parietal* and *frontal* signal. However, that claim is true only if we
# have some data from the same subject already acquired.
#
# The problem is that we split the data randomly into 5 folds (default, see
# :func:`.run_cross_validation`). This means that data from one subject could
# be both in the training and the testing set. If this is the case, then the
# model can learn the subjects' specific characteristics and apply it to the
# testing set. Thus, it is not true that we can decode it for an unseen
# subject, but for an unseen timepoint for a subject that for whom we already
# have data.
#
# To test for unseen subject, we need to make sure that all the data from each
# subject is either on the training or the testing set, but not in both.
#
# We can use ``scikit-learn``'s
# :class:`sklearn.model_selection.GroupShuffleSplit` and specify which is the
# grouping column using the ``group`` parameter.
# By setting ``return_estimator="final"``, the :func:`.run_cross_validation`
# function returns the estimator fitted with all the data. We will use this
# later to do some analyses.
cv = GroupShuffleSplit(n_splits=5, test_size=0.5, random_state=42)

scores, model = run_cross_validation(
    X=X,
    y=y,
    data=df_fmri,
    preprocess="zscore",
    model="svm",
    cv=cv,
    groups="subject",
    problem_type="classification",
    return_estimator="final",
)

print(scores["test_score"].mean())

###############################################################################
# After testing on independent subjects, we can now claim that given a new
# subject, we can predict the kind of event.
#
# Let's do some visualization on how these two features interact and what
# the preprocessing part of the model does.

# Plot the raw features
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
sns.scatterplot(
    x="parietal", y="frontal", hue="event", data=df_fmri, ax=axes[0], s=5
)
axes[0].set_title("Raw data")

# Plot the preprocessed features
pre_X = preprocess(
    model, X=X, data=df_fmri, until="zscore", with_column_types=True
)

pre_df = pre_X.join(df_fmri[y])

sns.scatterplot(
    x="parietal__:type:__continuous",
    y="frontal__:type:__continuous",
    hue="event",
    data=pre_df,
    ax=axes[1],
    s=5,
)

axes[1].set_title("Preprocessed data")

###############################################################################
# In this case, the preprocessing is nothing more than a
# :class:`sklearn.preprocessing.StandardScaler`.
#
# It seems that the data is not quite linearly separable. Let's now visualize
# how the SVM does this complex task.

# Get the model from the pipeline
clf = model[2]
fig = plt.figure()
ax = sns.scatterplot(
    x="parietal__:type:__continuous",
    y="frontal__:type:__continuous",
    hue="event",
    data=pre_df,
    s=5,
)

xlim = ax.get_xlim()
ylim = ax.get_ylim()

# Create grid to evaluate model
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T

# Create pandas.DataFrame
xy_df = pd.DataFrame(
    data=xy,
    columns=["parietal__:type:__continuous", "frontal__:type:__continuous"],
)

Z = clf.decision_function(xy_df).reshape(XX.shape)
a = ax.contour(XX, YY, Z, colors="k", levels=[0], alpha=0.5, linestyles=["-"])
ax.set_title("Preprocessed data with SVM decision function boundaries")
