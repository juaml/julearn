# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL
import pandas as pd
import numpy as np

from julearn.transformers.confounds import DataFrameConfoundRemover


X = pd.DataFrame({'A': np.arange(10), 'B': np.arange(10, 20),
                  'cheese__:type:__confound': np.arange(50, 60)})

X_multi = pd.DataFrame({'A': np.arange(10), 'B': np.arange(10, 20),
                        'cookie__:type:__confound': np.arange(40, 50),
                        'cheese__:type:__confound': np.arange(50, 60)})


def test_DataFrameConfoundRemover_suffix_one_conf_no_error():

    remover = DataFrameConfoundRemover()
    X_trans = remover.fit_transform(X)
    actual = X_trans.columns
    expected = X[['A', 'B']].columns
    assert len(actual) == len(expected)
    assert all([a == b for a, b in zip(actual, expected)])


def test_DataFrameConfoundRemover_suffix_multi_conf_no_error():

    remover = DataFrameConfoundRemover()
    X_trans = remover.fit_transform(X_multi)
    actual = X_trans.columns
    expected = X[['A', 'B']].columns
    assert len(actual) == len(expected)
    assert all([a == b for a, b in zip(actual, expected)])


def test_DataFrameConfoundRemover_no_suffix_one_conf_str_no_error():

    remover = DataFrameConfoundRemover(confounds='cheese__:type:__confound')
    X_trans = remover.fit_transform(X)
    actual = X_trans.columns
    expected = X[['A', 'B']].columns
    assert len(actual) == len(expected)
    assert all([a == b for a, b in zip(actual, expected)])


def test_DataFrameConfoundRemover_no_suffix_one_conf_list_no_error():

    remover = DataFrameConfoundRemover(confounds=['cheese__:type:__confound'])
    X_trans = remover.fit_transform(X)
    actual = X_trans.columns
    expected = X[['A', 'B']].columns
    assert len(actual) == len(expected)
    assert all([a == b for a, b in zip(actual, expected)])


def test_DataFrameConfoundRemover_no_suffix_mutli_conf_list_no_error():

    remover = DataFrameConfoundRemover(
        confounds=['B', 'cheese__:type:__confound'])
    X_trans = remover.fit_transform(X)
    actual = X_trans.columns
    expected = X[['A']].columns
    assert len(actual) == len(expected)
    assert all([a == b for a, b in zip(actual, expected)])
