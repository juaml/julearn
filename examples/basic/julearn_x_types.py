# -*- coding: utf-8 -*-
"""
Customize your pipeline using X_types
=======================================================

In this example, we'll look at how you can individualize single analysis steps by assigning 
a "type" to a column (`X_types`), using the `penguins` example dataset.
"""

# Authors: Vincent Küppers <v.kueppers@fz-juelich.de>
#          Hanwen Bi <h.bi@fz-juelich.de>
# License: AGPL

from seaborn import load_dataset
from julearn.pipeline import PipelineCreator
from julearn import run_cross_validation
from julearn.utils import configure_logging


###############################################################################
# Set the logging level to info to see extra information
configure_logging(level="INFO")

###############################################################################
# Load dataset, remove rows with missing values
# define features + target

df_penguins = load_dataset("penguins")
df_penguins = df_penguins.dropna().reset_index(drop=True)
df_penguins = df_penguins.replace({"sex": {"Female": 1, "Male": 2}})

df_penguins.head()

X = df_penguins.iloc[:,2:,].columns.tolist()
y = "species"

###############################################################################
# Define custom types for columns (of input features, X).  
# We will use those types in the `PipelineCreator` to specify input.  
# To adress all features in the `PipelineCreator` use `*`.
# ! Important: if you define X types, you also need to be specific with
# `apply_to`. In the PipelineCreator you can set to which features your model
# (here `svm`) is applied to. If no input is given, all processing steps
# (including the final model) are applied to all *non* defined features
# (i.e. `continuous`). By default PCA output is of type:`continuous`.

X_types = {
    "bill": ["bill_length_mm", "bill_depth_mm"],
    "body": ["flipper_length_mm", "body_mass_g"],
    "our_confound": ["sex"]
}

creator_1 = PipelineCreator(problem_type="classification", apply_to="*")

creator_1.add("zscore", apply_to="*")
creator_1.add("pca", apply_to=["bill"], n_components=1)
creator_1.add("svm")

scores_1, model_1 = run_cross_validation(
            X=X, y=y, data=df_penguins, 
            X_types = X_types,
            model = creator_1, 
            return_estimator="final"
)
print(scores_1['test_score'])

###############################################################################
# We can also z-score by the X_types defined before. Additionally we will
# `minmaxscale` other variables.

creator_2 = PipelineCreator(problem_type="classification", apply_to="*")

creator_2.add("zscore", apply_to="bill")
creator_2.add("scaler_minmax", apply_to="body")
creator_2.add("remove_confound", apply_to=["bill", "body"], confounds=["our_confound"])
creator_2.add("svm")

scores_2, model_2 = run_cross_validation(
            X=X, y=y, data=df_penguins, 
            X_types = X_types,
            model = creator_2, 
            return_estimator="final"
)
print(scores_2['test_score'])

###############################################################################
# Now, let's compare both preprocessing pipelines.

from julearn.inspect import preprocess
help(preprocess)

###############################################################################
# By setting the parameter `until=` to pipeline step, you can track how the
# variables were transformed until that step (including).

print('variables before pipeline \n', df_penguins[X].head())
print('variables after zscore \n', preprocess(model_1, X, df_penguins, until='zscore').head())
print('variables after pca \n', preprocess(model_1, X, df_penguins, until='pca').head())

###############################################################################
# We can also see how it looks like after remove confounds.

print('variables before pipeline \n', df_penguins[X].head())
print('variables after zscore \n', preprocess(model_2, X, df_penguins, until='zscore').head())
print('variables after pca \n', preprocess(model_2, X, df_penguins, until='remove_confound').head())