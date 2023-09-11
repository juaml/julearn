# Authors: Vera Komeyer <v.komeyer@fz-juelich.de>
#          Fede Raimondo <f.raimondo@fz-juelich.de>
# License: AGPL

"""
Model Comparison
================

In the previous section, we saw how to evaluate a single model using
cross-validation. The example model seems to perform decently well. However,
how do we know that it can't be better? Building machine-learning models is
always a matter of *benchmarking*. We want to know how well our model performs,
compared to other models. We already saw how to evaluate a model's performance
using cross-validation. This is a good start, but it is not enough. We can use
cross-validation to evaluate the performance of a single model, but we can't use
it to compare different models. We could build different models and evaluate them
using cross-validation, but then we would have to compare the results manually.
This is not only tedious, but also error-prone. We need a way to compare
different models in a statistically sound way.

To statistically compare different models, ``julearn`` provides a built-in
corrected ``t-test``. To see how to apply it, we will first build three
different models, each with different learning algorithms.

To perform a binary classification (and not a multi-class classification) we
will switch to the :func:`breast cancer dataset from scikit-learn
<sklearn.datasets.load_breast_cancer>` as an example. The target to be
predicted is if the cancer is malignant or benign.
"""

import pandas as pd
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target

X = data.feature_names.tolist()
y = "target"
X_types = {"continuous": [".*"]}

# sphinx_gallery_start_ignore
pd.set_option("display.max_columns", 9)
# sphinx_gallery_end_ignore

df.head()

###############################################################################
# We will use the same cross-validation splitter as in the previous section
# and two scorers: ``accuracy`` and ``roc_auc``.

from sklearn.model_selection import RepeatedStratifiedKFold

cv_splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)

scoring = ["accuracy", "roc_auc"]

###############################################################################
# We use three different learning algorithms to build three different models.
# We will use the default hyperparameters for each of them.
#
# **Model 1**: default SVM.

from julearn.pipeline import PipelineCreator
from julearn import run_cross_validation

creator1 = PipelineCreator(problem_type="classification")
creator1.add("zscore")
creator1.add("svm")

scores1 = run_cross_validation(
    X=X,
    y=y,
    X_types=X_types,
    data=df,
    model=creator1,
    scoring=scoring,
    cv=cv_splitter,
)

###############################################################################
# **Model 2**: default Random Forest.
#
# As we can see in :ref:`available_models`, we can use the ``"rf"`` string to
# use a random forest.

creator2 = PipelineCreator(problem_type="classification")
creator2.add("zscore")
creator2.add("rf")

scores2 = run_cross_validation(
    X=X,
    y=y,
    X_types=X_types,
    data=df,
    model=creator2,
    scoring=scoring,
    cv=cv_splitter,
)

###############################################################################
# **Model 3**: default Logistic Regression.

creator3 = PipelineCreator(problem_type="classification")
creator3.add("zscore")
creator3.add("logit")

scores3 = run_cross_validation(
    X=X,
    y=y,
    X_types=X_types,
    data=df,
    model=creator3,
    scoring=scoring,
    cv=cv_splitter,
)

###############################################################################
# We will add a column to each scores DataFrames to be able to use names to
# identify the models later on.

scores1["model"] = "svm"
scores2["model"] = "rf"
scores3["model"] = "logit"

# sphinx_gallery_start_ignore
# The following lines are only meant for the documentation to work and not
# needed for the example to run. This will be removed as soon as sphix-gallery
# is able to hide code blocks.
scores1.to_csv("/tmp/doc_scores1.csv")
scores2.to_csv("/tmp/doc_scores2.csv")
scores3.to_csv("/tmp/doc_scores3.csv")
# sphinx_gallery_end_ignore

###############################################################################
# Statistical comparisons
# -----------------------
#
# Comparing the performance of these three models is now as easy as
# the following one-liner:

from julearn.stats.corrected_ttest import corrected_ttest

stats_df = corrected_ttest(scores1, scores2, scores3)

###############################################################################
# This gives us a DataFrame with the corrected t-test results for each pairwise
# comparison of the three models' test scores:
#
# We can see, that none of the models performed better with respect to
# neither accuracy nor roc_auc.

print(stats_df)

###############################################################################
# Score visualizations
# --------------------
#
# Visualizations can help to get a better intuitive understanding of the
# differences between the models. To get a better overview of the performances
# of our three models, we can make use of ``julearn``'s visualization tool to
# plot the scores in an interactive manner. As visualizations are not part of the
# core functionality of ``julearn``, you will need to first manually
# **install the additional visualization dependencies**.
#
# From here we can create the interactive plot. Interactive, because you can
# choose the models to be displayed and the scorer to be plotted.

from julearn.viz import plot_scores

panel = plot_scores(scores1, scores2, scores3)
# panel.show()
# uncomment the previous line show the plot
# read the documentation for more information
#  https://panel.holoviz.org/getting_started/build_app.html#deploying-panels

###############################################################################
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

###############################################################################
# Well done, you made it until here and are now ready to dive into
# :ref:`selected_deeper_topics`!
# Maybe you are curious to learn :ref:`confound_removal` or want to learn more
# about :ref:`model_inspection`.
