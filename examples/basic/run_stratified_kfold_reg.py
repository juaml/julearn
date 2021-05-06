"""
repeated_stratified_splits
stratified_splits
============================

This demo example uses the 'diabetes' data from sklearn datasets to 
perform repeated_stratified_splits/repeated_stratified_splits for 
regression problem

"""
# Authors: Shammi More <s.more@fz-juelich.de>
#          Federico Raimondo <f.raimondo@fz-juelich.de>
#
# License: AGPL

import math
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold

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
data_df = pd.concat([features, target], axis=1)

X = ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
y = 'target'

###############################################################################
# Define number of splits, number of repeats and random seed
num_splits = 5 
num_repeats = 5 
rand_seed = 200

num_bins = math.floor(len(data_df)/num_splits) # num of bins to be created
bins_on = data_df.target # variable to be used for stratification
qc = pd.cut(bins_on.tolist(), num_bins)  # divides data in bins

###############################################################################
# StratifiedKFold for regression
cv = StratifiedKFold(n_splits=num_splits, shuffle=False, random_state=None)
for train_index, test_index in cv.split(data_df, qc.codes):
    print('test_index', test_index)
    print('len of test and train', len(test_index), len(train_index))

###############################################################################
RepeatedStratifiedKFold for regression
cv = RepeatedStratifiedKFold(n_splits=num_splits, n_repeats=num_repeats, random_state=rand_seed)
for train_index, test_index in cv.split(data_df, qc.codes):
    print('test_index', test_index)
    print('len of test and train', len(test_index), len(train_index))






    

    
     

