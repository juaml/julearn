"""
Regression Analysis
============================

This example uses the 'diabetes' data from sklearn datasets and performs
a regression analysis using a Ridge Regression model.

"""

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from julearn import run_cross_validation
from julearn.utils import configure_logging

###############################################################################
# Set the logging level to info to see extra information
configure_logging(level='INFO')

###############################################################################
# load the diabetes data from sklearn as a pandas dataframe
features, target = load_diabetes(return_X_y=True, as_frame=True)

###############################################################################
# Dataset contains ten variables age, sex, body mass index, average  blood
# pressure, and six blood serum measurements (s1-s6) diabetes patients and
# a quantitative measure of disease progression one year after baseline which
# is the target we are interested in predicting.

print('Features: \n', features.head())
print('Target: \n', target.describe())

###############################################################################
# Let's combine features and target together in one dataframe and define X
# and y
data_diabetes = pd.concat([features, target], axis=1)

X = ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
y = 'target'

###############################################################################
# calculate correlations between the features/variables and plot it as heat map
corr = data_diabetes.corr()
fig, ax = plt.subplots(1, 1, figsize=(10, 7))
sns.set(font_scale=1.2)
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns,
            annot=True, fmt="0.1f")


###############################################################################
# Split the dataset into train and test
train_diabetes, test_diabetes = train_test_split(data_diabetes, test_size=0.3)

###############################################################################
# Train a ridge regression model on train dataset and use mean absolute error
# for scoring
scores, model = run_cross_validation(X=X, y=y, data=train_diabetes,
                                     problem_type='regression', model='ridge',
                                     return_estimator='final',
                                     scoring='neg_mean_absolute_error')


###############################################################################
# Mean value of mean absolute error across CV
print((scores['test_neg_mean_absolute_error'] * -1).mean())

###############################################################################
# Plot heatmap of mean absolute error (MAE) over all repeats and CV splits
n_repeats, n_splits = 5, 5
repeats = [str(i + 1) for i in range(0, n_repeats)]
splits = [str(i + 1) for i in range(0, n_splits)]
test_mae = scores['test_neg_mean_absolute_error'] * -1
test_mae = test_mae.reshape(n_repeats, n_splits)

df_mae = pd.DataFrame(test_mae, columns=splits, index=repeats)
df_mae.index.name = 'Repeats'
df_mae.columns.name = 'K-fold splits'

fig, ax = plt.subplots(1, 1, figsize=(10, 7))
sns.heatmap(df_mae, cmap="YlGnBu")
plt.title('Cross-validation MAE')


###############################################################################
# Let's plot the feature importance using the coefficients of the trained model

features = pd.DataFrame({'Features': X, 'importance': model['ridge'].coef_})
features.sort_values(by=['importance'], ascending=True, inplace=True)
features['positive'] = features['importance'] > 0

fig, ax = plt.subplots(1, 1, figsize=(10, 7))
features.set_index('Features', inplace=True)
features.importance.plot(kind='barh',
                         color=features.positive.map
                         ({True: 'blue', False: 'red'}))
ax.set(xlabel='Importance', title='Variable importance for Ridge Regression')


###############################################################################
# Use the final model to make predictions on test data and plot scatterplot
# of true values vs predicted values

y_true = test_diabetes[y]
y_pred = model.predict(test_diabetes[X])
mae = format(mean_absolute_error(y_true, y_pred), '.2f')
corr = format(np.corrcoef(y_pred, y_true)[1, 0], '.2f')

fig, ax = plt.subplots(1, 1, figsize=(10, 7))
sns.set_style("darkgrid")
plt.scatter(y_true, y_pred)
plt.plot(y_true, y_true)
xmin, xmax = ax.get_xlim()
ymin, ymax = ax.get_ylim()
text = 'MAE: ' + str(mae) + '   CORR: ' + str(corr)
ax.set(xlabel='True values', ylabel='Predicted values')
plt.title('Actual vs Predicted')
plt.text(xmax - 0.01 * xmax, ymax - 0.01 * ymax, text, verticalalignment='top',
         horizontalalignment='right', fontsize=12)
plt.axis('scaled')
