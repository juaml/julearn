"""
Grouped CV
==========

This example uses the ``fMRI`` dataset and performs GroupKFold
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
# License: AGPL

# Importing the necessary Python libraries
import numpy as np

from seaborn import load_dataset
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

from julearn.utils import configure_logging
from julearn import run_cross_validation

###############################################################################
# Set the logging level to info to see extra information
configure_logging(level="INFO")

###############################################################################
# Dealing with Cross-Validation techniques
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
    model="rf",
    problem_type="classification",
)

print(scores["test_score"].mean())

###############################################################################
# Train classification model with stratification on data
cv_stratified = StratifiedGroupKFold(n_splits=2)
scores, model = run_cross_validation(
    X=X,
    y=y,
    data=df_fmri,
    groups="subject",
    model="rf",
    problem_type="classification",
    cv=cv_stratified,
    return_estimator="final",
)

print(scores["test_score"].mean())

###############################################################################
# Train classification model without stratification on data
cv = GroupKFold(n_splits=2)
scores, model = run_cross_validation(
    X=X,
    y=y,
    data=df_fmri,
    groups="subject",
    model="rf",
    problem_type="classification",
    cv=cv,
    return_estimator="final",
)

print(scores["test_score"].mean())
