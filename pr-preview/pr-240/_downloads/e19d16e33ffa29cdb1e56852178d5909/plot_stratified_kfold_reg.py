"""
Stratified K-fold CV for regression analysis
============================================

This example uses the ``diabetes`` data from ``sklearn datasets`` to
perform stratified Kfold CV for a regression problem,

.. include:: ../../links.inc
"""
# Authors: Shammi More <s.more@fz-juelich.de>
#          Federico Raimondo <f.raimondo@fz-juelich.de>
#          Leonard Sasse <l.sasse@fz-juelich.de>
# License: AGPL

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import KFold

from julearn import run_cross_validation
from julearn.utils import configure_logging
from julearn.model_selection import ContinuousStratifiedKFold

###############################################################################
# Set the logging level to info to see extra information.
configure_logging(level="INFO")

###############################################################################
# Load the diabetes data from ``sklearn`` as a ``pandas.DataFrame``.
features, target = load_diabetes(return_X_y=True, as_frame=True)

###############################################################################
# Dataset contains ten variables age, sex, body mass index, average blood
# pressure, and six blood serum measurements (s1-s6) diabetes patients and
# a quantitative measure of disease progression one year after baseline which
# is the target we are interested in predicting.

print("Features: \n", features.head())
print("Target: \n", target.describe())

###############################################################################
# Let's combine features and target together in one dataframe and create some
# outliers to see the difference in model performance with and without
# stratification.

data_df = pd.concat([features, target], axis=1)

# Create outliers for test purpose
new_df = data_df[(data_df.target > 145) & (data_df.target <= 150)]
new_df["target"] = [590, 580, 597, 595, 590, 590, 600]
data_df = pd.concat([data_df, new_df], axis=0)
data_df = data_df.reset_index(drop=True)

# Define X, y
X = ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]
y = "target"


###############################################################################
# Define number of bins/group for stratification. The idea is that each "bin"
# will be equally represented in each fold. The number of bins should be
# chosen such that each bin has a sufficient number of samples so that each
# fold has more than one sample from each bin.
# Let's see a couple of histrograms with different number of bins.

sns.displot(data_df, x="target", bins=60)

sns.displot(data_df, x="target", bins=40)

sns.displot(data_df, x="target", bins=20)

###############################################################################
# From the histogram above, we can see that the data is not uniformly
# distributed. We can see that the data is skewed towards the lower end of
# the target variable. We can also see that there are some outliers in the
# data. In any case, even with a low number of splits, some groups will not be
# represented in each fold. Lets continue with 40 bins which gives a good
# granularity.

cv_stratified = ContinuousStratifiedKFold(n_bins=40, n_splits=5, shuffle=False)

###############################################################################
# Train a linear regression model with stratification on target.

scores_strat, model = run_cross_validation(
    X=X,
    y=y,
    data=data_df,
    preprocess="zscore",
    cv=cv_stratified,
    problem_type="regression",
    model="linreg",
    return_estimator="final",
    scoring="neg_mean_absolute_error",
)

###############################################################################
# Train a linear regression model without stratification on target.

cv = KFold(n_splits=5, shuffle=False, random_state=None)
scores, model = run_cross_validation(
    X=X,
    y=y,
    data=data_df,
    preprocess="zscore",
    cv=cv,
    problem_type="regression",
    model="linreg",
    return_estimator="final",
    scoring="neg_mean_absolute_error",
)

###############################################################################
# Now we can compare the test score for model trained with and without
# stratification. We can combine the two outputs as ``pandas.DataFrame``.

scores_strat["model"] = "With stratification"
scores["model"] = "Without stratification"
df_scores = scores_strat[["test_score", "model"]]
df_scores = pd.concat([df_scores, scores[["test_score", "model"]]])

###############################################################################
# Plot a boxplot with test scores from both the models. We see here that
# the test score is higher when CV splits were not stratified.

fig, ax = plt.subplots(1, 1, figsize=(10, 7))
sns.set_style("darkgrid")
ax = sns.boxplot(x="model", y="test_score", data=df_scores)
ax = sns.swarmplot(x="model", y="test_score", data=df_scores, color=".25")
