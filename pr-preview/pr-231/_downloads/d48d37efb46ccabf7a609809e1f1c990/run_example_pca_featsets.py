"""
Regression Analysis
===================

This example uses the ``diabetes`` data from ``sklearn datasets`` and performs
a regression analysis using a Ridge Regression model. We'll use the
``julearn.PipelineCreator`` to create a pipeline with two different PCA steps and
reduce the dimensionality of the data, each one computed on a different
subset of features.

"""
# Authors: Georgios Antonopoulos <g.antonopoulos@fz-juelich.de>
#          Kaustubh R. Patil <k.patil@fz-juelich.de>
#          Shammi More <s.more@fz-juelich.de>
#          Federico Raimondo <f.raimondo@fz-juelich.de>
# License: AGPL

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from julearn import run_cross_validation
from julearn.utils import configure_logging
from julearn.pipeline import PipelineCreator
from julearn.inspect import preprocess

###############################################################################
# Set the logging level to info to see extra information.
configure_logging(level="INFO")

###############################################################################
# Load the diabetes data from ``sklearn`` as a ``pandas.DataFrame``.
features, target = load_diabetes(return_X_y=True, as_frame=True)

###############################################################################
# Dataset contains ten variables age, sex, body mass index, average  blood
# pressure, and six blood serum measurements (s1-s6) diabetes patients and
# a quantitative measure of disease progression one year after baseline which
# is the target we are interested in predicting.

print("Features: \n", features.head())
print("Target: \n", target.describe())

###############################################################################
# Let's combine features and target together in one dataframe and define X
# and y
data_diabetes = pd.concat([features, target], axis=1)

X = ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]
y = "target"

###############################################################################
# Assign types to the features and create feature groups for PCA.
# We will keep 1 component per PCA group.
X_types = {
    "pca1": ["age", "bmi", "bp"],
    "pca2": ["s1", "s2", "s3", "s4", "s5", "s6"],
    "categorical": ["sex"],
}

###############################################################################
# Create a pipeline to process the data and the fit a model. We must specify
# how each ``X_type`` will be used. For example if in the last step we do not
# specify ``apply_to=["continuous", "categorical"]``, then the pipeline will not
# know what to do with the categorical features.
creator = PipelineCreator(problem_type="regression")
creator.add("pca", apply_to="pca1", n_components=1, name="pca_feats1")
creator.add("pca", apply_to="pca2", n_components=1, name="pca_feats2")
creator.add("ridge", apply_to=["continuous", "categorical"])

###############################################################################
# Split the dataset into train and test.
train_diabetes, test_diabetes = train_test_split(data_diabetes, test_size=0.3)

###############################################################################
# Train a ridge regression model on train dataset and use mean absolute error
# for scoring.
scores, model = run_cross_validation(
    X=X,
    y=y,
    X_types=X_types,
    data=train_diabetes,
    model=creator,
    scoring="r2",
    return_estimator="final",
)

###############################################################################
# The scores dataframe has all the values for each CV split.
print(scores.head())

###############################################################################
# Mean value of mean absolute error across CV.
print(scores["test_score"].mean())

###############################################################################
# Let's see how the data looks like after preprocessing. We will process the
# data until the first PCA step. We should get the first PCA component for
# ["age", "bmi", "bp"] and leave other features untouched.
data_processed1 = preprocess(model, X, data=train_diabetes, until="pca_feats1")
print("Data after preprocessing until PCA step 1")
data_processed1.head()

###############################################################################
# We will process the data until the second PCA step. We should now also get
# one PCA component for ["s1", "s2", "s3", "s4", "s5", "s6"].
data_processed2 = preprocess(model, X, data=train_diabetes, until="pca_feats2")
print("Data after preprocessing until PCA step 2")
data_processed2.head()

###############################################################################
# Now we can get the MAE fold and repetition:
df_mae = scores.set_index(["repeat", "fold"])["test_score"].unstack() * -1
df_mae.index.name = "Repeats"
df_mae.columns.name = "K-fold splits"

print(df_mae)

###############################################################################
# Plot heatmap of mean absolute error (MAE) over all repeats and CV splits.
fig, ax = plt.subplots(1, 1, figsize=(10, 7))
sns.heatmap(df_mae, cmap="YlGnBu")
plt.title("Cross-validation MAE")

###############################################################################
# Use the final model to make predictions on test data and plot scatterplot
# of true values vs predicted values.
y_true = test_diabetes[y]
y_pred = model.predict(test_diabetes[X])
mae = format(mean_absolute_error(y_true, y_pred), ".2f")
corr = format(np.corrcoef(y_pred, y_true)[1, 0], ".2f")

fig, ax = plt.subplots(1, 1, figsize=(10, 7))
sns.set_style("darkgrid")
plt.scatter(y_true, y_pred)
plt.plot(y_true, y_true)
xmin, xmax = ax.get_xlim()
ymin, ymax = ax.get_ylim()
text = "MAE: " + str(mae) + "   CORR: " + str(corr)
ax.set(xlabel="True values", ylabel="Predicted values")
plt.title("Actual vs Predicted")
plt.text(
    xmax - 0.01 * xmax,
    ymax - 0.01 * ymax,
    text,
    verticalalignment="top",
    horizontalalignment="right",
    fontsize=12,
)
plt.axis("scaled")
