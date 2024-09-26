"""
Working with ``pandas``
=======================

This example uses the ``fmri`` dataset to transform and combine data in order
to prepare it to be used by ``julearn``.


References
----------

  Waskom, M.L., Frank, M.C., Wagner, A.D. (2016). Adaptive engagement of
  cognitive control in context-dependent decision-making. Cerebral Cortex.

.. include:: ../../links.inc
"""
# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#
# License: AGPL

from seaborn import load_dataset
import pandas as pd

###############################################################################
# One of the key elements that make ``julearn`` easy to use, is the possibility
# to work directly with ``pandas.DataFrame``, similar to MS Excel spreadsheets
# or csv files.
#
# Ideally, we will have everything tabulated and organised for ``julearn``, but
# it might not be your case. You might have some files with the fMRI values, some
# others with demographics, some other with diagnostic metrics or behavioral
# results.
#
# You need to prepare these files for ``julearn``.
#
# One option is to manually edit the files and make sure that everything is
# ready to do some machine-learning. However, this is error-prone.
#
# Fortunately, `pandas`_ provides several tools to deal with this task.
#
# This example is a collection of some of these useful methods.
#
# Let's start with the ``fmri`` dataset.

df_fmri = load_dataset("fmri")

###############################################################################
# Let's see what this dataset has.
#
df_fmri.head()

###############################################################################
# From long to wide format
# We have seen this in other examples. If we want to use julearn, each feature
# must be a columns. In order to use the signals from different regions as
# ~~~~~~~~~~~~~~~~~~~~~~~~
# features, we need to convert this dataframe from the long format to the wide
# format.
#
# We will use the ``pivot`` method.
df_fmri = df_fmri.pivot(
    index=["subject", "timepoint", "event"], columns="region", values="signal"
)

###############################################################################
# This method reshapes the table, keeping the specified elements as index,
# columns and values.
#
# In our case, the values are extracted from the *signal* column. The columns
# from the *region* column and *subject*, *timepoint* and *event* becomes the
# index.
#
# The index is what identifies each sample. As a rule, the index can't be
# duplicated. If each subject has more than one timepoint, and each timepoint
# has more than one event, then these 3 elements are needed as the index.
#
# Let's see what we have here:
df_fmri.head()

###############################################################################
# Now this is in the format we want. However, in order to access the index
# as columns ``df_fmri["subject"]`` we need to reset the index.
#
# Check the subtle but important difference:
df_fmri = df_fmri.reset_index()
df_fmri.head()

###############################################################################
# Merging or joining ``DataFrame``
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# So now we have our fMRI data tabulated for ``julearn``. However, it might be
# the case that we have some important information in another file. For example,
# the subjects' age and the place where they were scanned.
#
# For the purpose of the example, we'll create the dataframe here.
metadata = {
    "subject": [f"s{i}" for i in range(14)],
    "age": [23, 21, 31, 29, 43, 23, 43, 28, 48, 29, 35, 23, 34, 25],
    "scanner": ["a"] * 6 + ["b"] * 8,
}
df_meta = pd.DataFrame(metadata)
df_meta

###############################################################################
# We will use the ``join`` method. This method will join the two dataframes,
# matching elements by the *index*.
#
# In this case, the matching element (or index) will be the column ``subject``.
# We need to set the index in each dataframe before join.
df_fmri = df_fmri.set_index("subject")
df_meta = df_meta.set_index("subject")
df_fmri = df_fmri.join(df_meta)
df_fmri

###############################################################################
# Finally, let's reset the index and have it ready for ``julearn``.
df_fmri = df_fmri.reset_index()

###############################################################################
# Now we can use, for example, *age* and *scanner* as confounds.

###############################################################################
# Reshaping data frames (more complex)
# Lets suppose that our prediction target is now the *age* and we want to use
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# as features the frontal and parietal value during each event. For this
# purpose, we need to convert the event values into columns. There are two
# events: *cue* and *stim*. So this will result in 4 columns.
#
# We will still use the pivot, but in this case, we will have two values:
df_fmri = df_fmri.pivot(
    index=["subject", "timepoint", "age", "scanner"],
    columns="event",
    values=["frontal", "parietal"],
)
df_fmri

###############################################################################
# Since the column names are combinations of the values in the *event* column
# and the previous *frontal* and *parietal* columns, it is now a multi-level
# column name.
df_fmri.columns

###############################################################################
# The following trick will join the different levels using an underscore (*_*)
df_fmri.columns = ["_".join(x) for x in df_fmri.columns]
df_fmri

###############################################################################
# We have finally the information we want. We can now reset the index.
df_fmri = df_fmri.reset_index()
