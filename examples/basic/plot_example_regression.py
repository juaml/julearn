"""
Regression Analysis
============================

This example uses the 'diabetes' dataset from sklearn datasets and performs a regression analysis
using a Ridge Regression model.

"""

import pandas as pd
import seaborn as sns
import  numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

import matplotlib.pyplot as plt
import seaborn as sns
from seaborn import load_dataset

from julearn import run_cross_validation
from julearn.utils import configure_logging

###############################################################################
# Set the logging level to info to see extra information
configure_logging(level='INFO')

###############################################################################
# load the diabetes data from sklearn as a pandas dataframe
features, target = load_diabetes(return_X_y=True, as_frame=True)
data_diabetes = pd.concat([features, target], axis=1)
# data_diabetes, test_diabetes = train_test_split(data, test_size=0.2)

X = ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
y = 'target'

###############################################################################
# calculate correlations between the features/variables and save it as heat map
corr = data_diabetes.corr()
fig, ax = plt.subplots(1, 1, figsize=(10,7))
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns,  annot=True, fmt="0.1f")

###############################################################################
# Run Ridge regression model with different scoring functions (mean squared error, mean absolute error)
scores, model = run_cross_validation(X=X, y=y, data=data_diabetes, problem_type='regression', model='ridge', return_estimator='final', scoring=['neg_mean_squared_error', 'neg_mean_absolute_error'])
trained_model = model['ridge']
print((scores['test_neg_mean_absolute_error']*-1).mean())
print((scores['test_neg_mean_squared_error']*-1).mean())

###############################################################################
# Plot heatmap of MAE over all repeats and CV splits
n_repeats, n_splits = 5, 5
repeats = ['repeat ' + str(i + 1) for i in range(0, n_repeats)]
splits = ['split_' + str(i + 1) for i in range(0, n_splits)]
test_mae = scores['test_neg_mean_absolute_error']*-1
test_mae = test_mae.reshape(n_repeats, n_splits)

df_acc =pd.DataFrame(test_mae, columns=splits, index=repeats)
fig, ax = plt.subplots(1, 1, figsize=(10,7))
sns.heatmap(df_acc, cmap="YlGnBu")
sns.set(font_scale=1.2)
plt.title('Cross-validation MAE')
plt.savefig('CV_MAE_reg.png')

###############################################################################
# Use model to make predictions and plot scatter plot of actual values vs predicted values
y_true = data_diabetes[y]
y_pred = model.predict(data_diabetes[X])
mae = format(mean_absolute_error(y_true, y_pred), '.2f')
mse= format(mean_squared_error(y_true, y_pred), '.2f')
corr = format(np.corrcoef(y_pred, y_true)[1, 0], '.2f')
print(mae, mse, corr)

fig, ax = plt.subplots(1, 1, figsize=(10,7))
sns.set_style("darkgrid")
plt.scatter(y_true, y_pred)
plt.plot(y_true, y_true)
xmin, xmax = ax.get_xlim()
ymin, ymax = ax.get_ylim()
text = 'MAE: '+ str(mae)+ '   CORR: '+ str(corr)
ax.set(xlabel='True values', ylabel='Predicted values', title='Actual vs Predicted')
plt.text(xmax-.01*xmax, ymax-.01*ymax, text, verticalalignment='top', horizontalalignment='right', fontsize=12)
plt.axis('scaled')
sns.set(font_scale=1.2)
plt.savefig('actual_vs_predicted.png')
plt.close()

###############################################################################
# Let's plot the feature importance using the coefficents of trained model
features = pd.DataFrame({'Features': X, 'importance': trained_model.coef_})
features.sort_values(by=['importance'], ascending=True, inplace=True)
features['positive'] = features['importance'] > 0

fig, ax = plt.subplots(1, 1, figsize=(10,7))
features.set_index('Features', inplace=True)
features.importance.plot(kind='barh', color=features.positive.map({True: 'blue', False: 'red'}))
sns.set(font_scale=1.2)
ax.set(xlabel='Importance', title='Variable importance for Ridge Regression')
plt.savefig('feature_importance.png')

###############################################################################
# Compare with confound removal prediction vs after confound removal
X = ['age', 'sex', 'bp', 's1', 's2', 's3', 's4', 's5', 's6'] # remove bmi from features abd use  as confound
y = 'target'
confounds = 'bmi'

###############################################################################
# run without confound removal
scores_ncr, model_ncr = run_cross_validation(X=X, y=y, confounds=None, data=data_diabetes, problem_type='regression',
                                       model='ridge', preprocess_X=['zscore', 'pca'],
                                       return_estimator='final', scoring=['neg_mean_squared_error',
                                                                       'neg_mean_absolute_error'])

###############################################################################
# run with confound removal
scores_cr, model_cr = run_cross_validation(X=X, y=y, confounds=confounds, data=data_diabetes, problem_type='regression',
                                       model='ridge', preprocess_X=['zscore', 'pca'],
                                       return_estimator='final', scoring=['neg_mean_squared_error',
                                                                       'neg_mean_absolute_error'])

scores_df = pd.DataFrame({'NCR': scores_ncr['test_neg_mean_absolute_error']*-1, 'CVCR': scores_cr['test_neg_mean_absolute_error']*-1})
fig, ax = plt.subplots(1, 1, figsize=(10,7))
sns.set_style("darkgrid")
sns.boxplot(data=scores_df, width=0.5, palette="colorblind")
sns.stripplot(data=scores_df, jitter=True, dodge=True, marker='o', alpha=0.6, color='k')
ax.set(ylabel='Mean absolute error', title='MAE')
sns.set(font_scale=1.2)
plt.savefig('NCR_vs_CVCR.png')















