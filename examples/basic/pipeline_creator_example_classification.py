
"""
Pipeline creator examples
============================

In this example we will customize our pipeline using `PipelineCreator`.

"""
# Authors: Vincent Küppers <v.kueppers@fz-juelich.de>
#          Hanwen Bi <h.bi@fz-juelich.de>
#
# License: AGPL
# %%
from seaborn import load_dataset
from julearn.pipeline import PipelineCreator
from julearn import run_cross_validation
from julearn.inspect import preprocess

###############################################################################
# load dataset, we will use the `penguins` dataset, remove the
# rows with missing values and define features + target
df_penguins = load_dataset("penguins")
df_penguins = df_penguins.dropna().reset_index(drop=True)
df_penguins = df_penguins.replace({"sex": {"Female": 1, "Male": 2}})

df_penguins.head()

X = df_penguins.iloc[:,2:,].columns.tolist()
y = "species"


# %%
###############################################################################
# Define custom types (_bill_, __) for columns (input features)
# all column types that are not defined are `continous`
X_types = {
    "bill": ["bill_length_mm", "bill_depth_mm"],
    "body": ["flipper_length_mm", "body_mass_g"],
    "confound": ["sex"]
}

creator = PipelineCreator(problem_type="classification")

creator.add("zscore", apply_to="bill")
creator.add("scaler_minmax", apply_to="body")
creator.add("remove_confound", apply_to=["bill", "body"], confounds=["confound"])
creator.add("pca", apply_to=["bill"], n_components=1)
creator.add("svm")

scores, model = run_cross_validation(
            X=X, y=y, data=df_penguins,
            X_types = X_types,
            model = creator,
            return_estimator="final"
)
scores
