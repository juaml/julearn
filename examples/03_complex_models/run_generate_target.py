"""
Target Generation
=================

This example uses the ``iris`` dataset and tests a regression model in which
the target variable is generated from some features within the cross-validation
procedure. We will use the Iris dataset and generate a target variable using
PCA on the petal features. Then, we will evaluate if a regression model can
predict the generated target from the sepal features

.. include:: ../../links.inc
"""
# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
# License: AGPL

# %%
from seaborn import load_dataset
from julearn import run_cross_validation
from julearn.pipeline import PipelineCreator
from julearn.utils import configure_logging

###############################################################################
# Set the logging level to info to see extra information.
configure_logging(level="DEBUG")

###############################################################################
df_iris = load_dataset("iris")


###############################################################################
# As features, we will use the sepal length, width and petal length.
# We will try to predict the species.

X = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
y = "__generated__"  # to indicate to julearn that the target will be generated



# Define our feature types
X_types = {
    "sepal": ["sepal_length", "sepal_width"],
    "petal": ["petal_length", "petal_width"],
}

# %% reate the pipeline that will generate the features
target_creator = PipelineCreator(problem_type="transformer", apply_to="petal")
target_creator.add("pca", n_components=2)
target_creator.add("pick_columns", keep="pca__pca0")


# %% Create the final pipeline
creator = PipelineCreator(problem_type="regression")
creator.add("generate_target", apply_to="petal", transformer=target_creator)
creator.add(
    "linreg", apply_to="sepal",
)

# %%
scores, model = run_cross_validation(
    X=X,
    y=y,
    X_types=X_types,
    data=df_iris,
    model=creator,
    return_estimator="final",
    cv=2,
)

print(scores["test_score"])  # type: ignore

# %%
print(model)
