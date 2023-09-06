"""
Multiclass Classification
=========================

This example uses the ``iris`` dataset and performs multiclass
classification using a Support Vector Machine classifier and plots
heatmaps for cross-validation accuracies and plots confusion matrix
for the test data.

"""
# Authors: Shammi More <s.more@fz-juelich.de>
#          Federico Raimondo <f.raimondo@fz-juelich.de>
# License: AGPL

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from seaborn import load_dataset
from sklearn.model_selection import train_test_split, RepeatedKFold
from sklearn.metrics import confusion_matrix

from julearn import run_cross_validation
from julearn.utils import configure_logging

###############################################################################
# Set the logging level to info to see extra information
configure_logging(level="INFO")

###############################################################################
# load the iris data from seaborn
df_iris = load_dataset("iris")
X = ["sepal_length", "sepal_width", "petal_length"]
y = "species"

###############################################################################
# Split the dataset into train and test
train_iris, test_iris = train_test_split(
    df_iris, test_size=0.2, stratify=df_iris[y], random_state=200
)

###############################################################################
# We want to perform multiclass classification as iris dataset contains 3 kinds
# of species. We will first zscore all the features and then train a support
# vector machine classifier.

cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=200)
scores, model_iris = run_cross_validation(
    X=X,
    y=y,
    data=train_iris,
    model="svm",
    preprocess="zscore",
    problem_type="classification",
    cv=cv,
    scoring=["accuracy"],
    return_estimator="final",
)

###############################################################################
# The scores dataframe has all the values for each CV split.

scores.head()

###############################################################################
# Now we can get the accuracy per fold and repetition:

df_accuracy = scores.set_index(["repeat", "fold"])["test_accuracy"].unstack()
df_accuracy.index.name = "Repeats"
df_accuracy.columns.name = "K-fold splits"
df_accuracy

###############################################################################
# Plot heatmap of accuracy over all repeats and CV splits
sns.set(font_scale=1.2)
fig, ax = plt.subplots(1, 1, figsize=(10, 7))
sns.heatmap(df_accuracy, cmap="YlGnBu")
plt.title("Cross-validation Accuracy")

###############################################################################
# We can also test our final model's accuracy and plot the confusion matrix
# for the test data as an annotated heatmap
y_true = test_iris[y]
y_pred = model_iris.predict(test_iris[X])
cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))

print(cm)

###############################################################################
# Now that we have our confusion matrix, let's build another matrix with
# annotations.
cm_sum = np.sum(cm, axis=1, keepdims=True)
cm_perc = cm / cm_sum.astype(float) * 100
annot = np.empty_like(cm).astype(str)
nrows, ncols = cm.shape
for i in range(nrows):
    for j in range(ncols):
        c = cm[i, j]
        p = cm_perc[i, j]
        if c == 0:
            annot[i, j] = ""
        else:
            s = cm_sum[i]
            annot[i, j] = "%.1f%%\n%d/%d" % (p, c, s)

###############################################################################
# Finally we create another dataframe with the confusion matrix and plot
# the heatmap with annotations.
cm = pd.DataFrame(cm, index=np.unique(y_true), columns=np.unique(y_true))
cm.index.name = "Actual"
cm.columns.name = "Predicted"

fig, ax = plt.subplots(1, 1, figsize=(10, 7))
sns.heatmap(cm, cmap="YlGnBu", annot=annot, fmt="", ax=ax)
plt.title("Confusion matrix")
