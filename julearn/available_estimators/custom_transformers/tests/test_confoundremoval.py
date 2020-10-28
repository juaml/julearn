from julearn.available_estimators.custom_transformers import (
    DataFrameConfoundRemover)


import pandas as pd
import numpy as np


X = pd.DataFrame({'A': np.arange(10), 'B': np.arange(10, 20),
                  'cheese__:type:__confound': np.arange(50, 60)})

X_multi = pd.DataFrame({'A': np.arange(10), 'B': np.arange(10, 20),
                        'cookie__:type:__confound': np.arange(40, 50),
                        'cheese__:type:__confound': np.arange(50, 60)})


def test_DataFrameConfoundRemover_suffix_one_conf_no_error():

    remover = DataFrameConfoundRemover()
    X_tans = remover.fit_transform(X)
    X_tans.columns = X[['A', 'B']].columns


def test_DataFrameConfoundRemover_suffix_multi_conf_no_error():

    remover = DataFrameConfoundRemover()
    X_tans = remover.fit_transform(X_multi)
    X_tans.columns = X[['A', 'B']].columns


def test_DataFrameConfoundRemover_no_suffix_one_conf_str_no_error():

    remover = DataFrameConfoundRemover(confounds='cheese__:type:__confound')
    X_tans = remover.fit_transform(X)
    X_tans.columns = X[['A', 'B']].columns


def test_DataFrameConfoundRemover_no_suffix_one_conf_list_no_error():

    remover = DataFrameConfoundRemover(confounds=['cheese__:type:__confound'])
    X_trans = remover.fit_transform(X)
    X_trans.columns = X[['A', 'B']].columns


def test_DataFrameConfoundRemover_no_suffix_mutli_conf_list_no_error():

    remover = DataFrameConfoundRemover(
        confounds=['B', 'cheese__:type:__confound'])
    X_trans = remover.fit_transform(X)
    X_trans.columns = X[['A']].columns
