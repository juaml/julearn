"""
Simple Model Comparison
=======================

This example uses the ``iris`` dataset and performs binary classifications
using different models. At the end, it compares the performance of the models
using different scoring functions and performs a statistical test to assess
whether the difference in performance is significant.

.. include:: ../../links.inc
"""
# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
# License: AGPL

from seaborn import load_dataset
from sklearn.model_selection import RepeatedStratifiedKFold
from julearn import run_cross_validation
from julearn.utils import configure_logging
from julearn.stats.corrected_ttest import corrected_ttest

###############################################################################
# Set the logging level to info to see extra information.
configure_logging(level="INFO")

###############################################################################
df_iris = load_dataset("iris")

###############################################################################
# The dataset has three kind of species. We will keep two to perform a binary
# classification.
df_iris = df_iris[df_iris["species"].isin(["versicolor", "virginica"])]

###############################################################################
# As features, we will use the sepal length, width and petal length.
# We will try to predict the species.

X = ["sepal_length", "sepal_width", "petal_length"]
y = "species"
scores = run_cross_validation(
    X=X,
    y=y,
    data=df_iris,
    model="svm",
    problem_type="classification",
    preprocess="zscore",
)

print(scores["test_score"])

###############################################################################
# Additionally, we can choose to assess the performance of the model using
# different scoring functions.
#
# For example, we might have an unbalanced dataset:

df_unbalanced = df_iris[20:]  # drop the first 20 versicolor samples
print(df_unbalanced["species"].value_counts())

###############################################################################
# So we will choose to use the `balanced_accuracy` and `roc_auc` metrics.

scoring = ["balanced_accuracy", "roc_auc"]

###############################################################################
# Since we are comparing the performance of different models, we will need
# to use the same random seed to split the data in the same way.

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)

###############################################################################
# First we will use a default SVM model.
scores1 = run_cross_validation(
    X=X,
    y=y,
    data=df_unbalanced,
    model="svm",
    preprocess="zscore",
    problem_type="classification",
    scoring=scoring,
    cv=cv,
)

scores1["model"] = "svm"

###############################################################################
# Second we will use a default Random Forest model.
scores2 = run_cross_validation(
    X=X,
    y=y,
    data=df_unbalanced,
    model="rf",
    preprocess="zscore",
    problem_type="classification",
    scoring=scoring,
    cv=cv,
)

scores2["model"] = "rf"

###############################################################################
# The third model will be a SVM with a linear kernel.
scores3 = run_cross_validation(
    X=X,
    y=y,
    data=df_unbalanced,
    model="svm",
    model_params={"svm__kernel": "linear"},
    preprocess="zscore",
    problem_type="classification",
    scoring=scoring,
    cv=cv,
)

scores3["model"] = "svm_linear"

###############################################################################
# We can now compare the performance of the models using corrected statistics.

stats_df = corrected_ttest(scores1, scores2, scores3)
print(stats_df)

###############################################################################
# .. rst-class:: hidden
#   This block is hidden in the documentation. This files are used to generate
#   the plots in the documentation. (not working for now)

# sphinx_gallery_start_ignore
# The following lines are only meant for the documentation to work and not
# needed for the example to run. This will be removed as soon as sphix-gallery
# is able to hide code blocks.
scores1.to_csv("/tmp/scores1.csv")
scores2.to_csv("/tmp/scores2.csv")
scores3.to_csv("/tmp/scores3.csv")
# sphinx_gallery_end_ignore

###############################################################################
# We can also plot the performance of the models using the ``julearn`` Score
# Viewer.

from julearn.viz import plot_scores

panel = plot_scores(scores1, scores2, scores3)
# panel.show()
# uncomment the previous line show the plot
# read the documentation for more information
#  https://panel.holoviz.org/getting_started/build_app.html#deploying-panels

###############################################################################
# This is how the plot looks like.
#
# .. note::
#    The plot is interactive. You can zoom in and out, and hover over.
#    However, buttons will not work in this documentation.
#
# .. bokeh-plot::
#    :source-position: none
#
#    from julearn.viz import plot_scores
#    from bokeh.io import output_notebook, show
#    import pandas as pd
#    output_notebook()
#    scores1 = pd.read_csv("/tmp/scores1.csv")
#    scores2 = pd.read_csv("/tmp/scores2.csv")
#    scores3 = pd.read_csv("/tmp/scores3.csv")
#    panel = plot_scores(scores1, scores2, scores3, width=600)
#    show(panel.get_root())
