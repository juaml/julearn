"""
Inspecting SVM models
=====================

This example uses the 'fmri' dataset, performs simple binary classification
using a Support Vector Machine classifier and analyse the model.


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
from julearn import run_cross_validation
from julearn.utils import configure_logging
from julearn.estimators import get_model

configure_logging(level='INFO')
df_iris = load_dataset('iris')
df_iris = df_iris[df_iris['species'].isin(['versicolor', 'virginica'])]

X = ['sepal_length', 'sepal_width', 'petal_length']
y = 'species'

ensemble_model = get_model('rf', problem_type='binary_classification',
                           n_estimators=100)

model = get_model('ds', problem_type='binary_classification',
                  ensemble=ensemble_model, algorithm='METADES')

scores = run_cross_validation(
    X=X, y=y, data=df_iris, model=model, preprocess_X='zscore')
print(scores['test_score'])
