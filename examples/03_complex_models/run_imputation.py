"""
Imputation of missing values
============================

This example uses the 'Diabetes' dataset with missing values created manually
and performs imputation using SimpleImputer, InterativeImputer and KNNImputer.

"""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Tsung-Hao Chen
# License: AGPL

import numpy as np
from sklearn.datasets import load_diabetes
from julearn.pipeline import PipelineCreator
from julearn.utils import configure_logging
from julearn import run_cross_validation

###############################################################################
# Set the logging level to info to see extra information
configure_logging(level="INFO")

###############################################################################
# Load in the data
df_features, target = load_diabetes(return_X_y=True, as_frame=True)

###############################################################################
# Dataset contains ten variables age, sex, body mass index, average blood
# pressure, and six blood serum measurements (s1-s6) diabetes patients and
# a quantitative measure of disease progression one year after baseline.

print("Features:\n", df_features.head())
print("Target:\n", target.describe())

###############################################################################
# Introduce missing values randomly in the dataset
rng = np.random.default_rng(seed=42)  # Reproducible results
missing_rate = 0.1  # 10% missing values
n_missing_samples = int(np.floor(missing_rate * df_features.size))

# Randomly select indices to introduce missing values
missing_indices = (
    rng.integers(0, df_features.shape[0], n_missing_samples),
    rng.integers(0, df_features.shape[1], n_missing_samples),
)
df_features.values[missing_indices] = np.nan

###############################################################################
# Let's define X and y. In addition, combine features and target together

X = ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]
y = "target"
X_types = {
    "continuous": [
        "age",
        "sex",
        "bmi",
        "bp",
        "s1",
        "s2",
        "s3",
        "s4",
        "s5",
        "s6",
    ]
}

data = df_features.copy()
data["target"] = target
###############################################################################
# The imputer_simple offers fundamental methods for handling missing values.
# These values can be replaced with a specified constant, or by using statistical
# measures such as the mean, median, or mode of each column containing the
# missing data.

pipeline_creator_1 = PipelineCreator(problem_type="regression")
pipeline_creator_1.add("imputer_simple", apply_to="*", strategy="mean")
pipeline_creator_1.add("ridge")


scores_1, model_1 = run_cross_validation(
    X=X,
    y=y,
    data=data,
    X_types=X_types,
    model=pipeline_creator_1,
    scoring="neg_mean_absolute_error",
    return_estimator="final",
)
print(scores_1)
###############################################################################
# A more advanced method is to use the imputer_iterative. This class models
# each feature with missing values as a function of the other features and
# uses these estimates to perform the imputation.
# max_iter specifies the maximum number of iterations that the imputer will perform.
# random_state controls the randomness of the imputation process.

pipeline_creator_2 = PipelineCreator(problem_type="regression")
pipeline_creator_2.add(
    "imputer_iterative", apply_to="*", max_iter=100, random_state=0
)
pipeline_creator_2.add("svm")


scores_2, model_2 = run_cross_validation(
    X=X,
    y=y,
    data=data,
    X_types=X_types,
    model=pipeline_creator_2,
    scoring="neg_mean_absolute_error",
    return_estimator="final",
)
print(scores_2)
###############################################################################
# The imputer_knn provides imputation for filling in missing values using the
# k-Nearest Neighbors approach.
# n_neighbors determines the number of neighboring samples to use for imputing each missing value.
# weights defines the weight function used in prediction, influencing how the neighbors' values are combined to impute the missing value.

pipeline_creator_3 = PipelineCreator(problem_type="regression")
pipeline_creator_3.add(
    "imputer_knn", apply_to="*", n_neighbors=2, weights="uniform"
)
pipeline_creator_3.add("ridge")


scores_3, model_3 = run_cross_validation(
    X=X,
    y=y,
    data=data,
    X_types=X_types,
    model=pipeline_creator_3,
    scoring="neg_mean_absolute_error",
    return_estimator="final",
)
print(scores_3)
