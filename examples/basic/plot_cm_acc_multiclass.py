"""
Multiclass Classification
============================

This example uses the 'iris' dataset and performs multiclass
classification using a Support Vector Machine classifier and plots
heatmaps for cross-validation accuracies and plots confusion matrix
for the test data.

"""

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from seaborn import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from julearn import run_cross_validation
from julearn.utils import configure_logging

###############################################################################
# Set the logging level to info to see extra information
configure_logging(level='INFO')

df_iris = load_dataset('iris')
X = ['sepal_length', 'sepal_width', 'petal_length']
y = 'species'

###############################################################################
# Split the dataset into train and test
train_iris, test_iris = train_test_split(df_iris, test_size=0.2,
                                         stratify=df_iris[y])

###############################################################################
# Perform multiclass classification as iris dataset contains 3 kinds of species
scores, model_iris = run_cross_validation(X=X, y=y, data=train_iris, model='svm',
                                          problem_type='multiclass_classification',
                                          scoring=['accuracy'],
                                          return_estimator='final')

###############################################################################
# Plot heatmap of accuracy over all repeats and CV splits
n_repeats, n_splits = 5, 5
repeats = [str(i + 1) for i in range(0,n_repeats)]
splits = [str(i + 1) for i in range(0,n_splits)]
test_acc = scores['test_accuracy'].reshape(n_repeats, n_splits)
df_acc = pd.DataFrame(test_acc, columns=splits, index=repeats)
df_acc.index.name = 'Repeats'
df_acc.columns.name = 'K-fold splits'

fig, ax = plt.subplots(1, 1, figsize=(10,7))
sns.heatmap(df_acc, cmap="YlGnBu")
sns.set(font_scale=1.2)
plt.title('Cross-validation Accuracy')

# plot confusion matrix for the test data as heatmap
y_true = test_iris[y]
y_pred = model_iris.predict(test_iris[X])
print('y_pred', y_pred)
cf_matrix = confusion_matrix(y_true, y_pred)

cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
cm_sum = np.sum(cm, axis=1, keepdims=True)
cm_perc = cm / cm_sum.astype(float) * 100
annot = np.empty_like(cm).astype(str)
nrows, ncols = cm.shape
for i in range(nrows):
    for j in range(ncols):
        c = cm[i, j]
        p = cm_perc[i, j]
        if c == 0:
            annot[i, j] = ''
        else:
            s = cm_sum[i]
            annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)

cm = pd.DataFrame(cm, index=np.unique(y_true), columns=np.unique(y_true))
cm.index.name = 'Actual'
cm.columns.name = 'Predicted'
fig, ax = plt.subplots(1, 1, figsize=(10,7))
sns.heatmap(cm, cmap="YlGnBu", annot=annot, fmt='', ax=ax)
sns.set(font_scale=1.2)
plt.title('Confusion matrix')



















