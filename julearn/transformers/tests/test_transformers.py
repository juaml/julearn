# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
# License: AGPL
from julearn.transformers.target import TargetTransfromerWrapper
from sklearn import svm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (StandardScaler, RobustScaler, MaxAbsScaler,
                                   MinMaxScaler, Normalizer,
                                   QuantileTransformer, PowerTransformer)
from sklearn.feature_selection import (GenericUnivariateSelect,
                                       SelectPercentile, SelectKBest,
                                       SelectFdr, SelectFpr, SelectFwe,
                                       VarianceThreshold)
from seaborn import load_dataset

import pytest

from julearn.utils.testing import do_scoring_test
from julearn.transformers import list_transformers, get_transformer

from julearn.transformers.tmp_transformers import (
    DropColumns, ChangeColumnTypes)

_features_transformers = {
    'zscore': StandardScaler,
    'scaler_robust': RobustScaler,
    'scaler_minmax': MinMaxScaler,
    'scaler_maxabs': MaxAbsScaler,
    'scaler_normalizer': Normalizer,
    'scaler_quantile': QuantileTransformer,
    'scaler_power': PowerTransformer,
    'select_univariate': GenericUnivariateSelect,
    'select_percentile': SelectPercentile,
    'select_k': SelectKBest,
    'select_fdr': SelectFdr,
    'select_fpr': SelectFpr,
    'select_fwe': SelectFwe,
    'select_variance': VarianceThreshold,
}

_transformer_params = {
    'scaler_quantile': {'n_quantiles': 10},
    'select_k': {'k': 2},
}


def test_feature_transformers():
    """Test transform X"""
    """Test simple binary classification"""
    df_iris = load_dataset('iris')

    # keep only two species
    df_iris = df_iris[df_iris['species'].isin(['setosa', 'virginica'])]
    X = ['sepal_length', 'sepal_width', 'petal_length']
    y = 'species'

    scorers = ['accuracy']

    for tr_name, tr_klass in _features_transformers.items():
        api_params = {'model': 'svm', 'preprocess_X': tr_name}
        if tr_name in _transformer_params:
            model_params = {f'{tr_name}__{k}': v
                            for k, v in _transformer_params[tr_name].items()}
            api_params['model_params'] = model_params
            tr = tr_klass(**_transformer_params[tr_name])
        else:
            tr = tr_klass()
        clf = make_pipeline(tr, svm.SVC())
        do_scoring_test(X, y, data=df_iris, api_params=api_params,
                        sklearn_model=clf, scorers=scorers)


def test_list_get_transformers():
    """Test list and getting transformers"""
    expected = list(_features_transformers.keys()) + [
        'pca',
        'remove_confound',
        'drop_columns',
        'change_column_types'
    ]
    actual = list_transformers()
    diff = set(actual) ^ set(expected)
    assert not diff

    expected = ['zscore', 'remove_confound']
    actual = list_transformers(target=True)
    diff = set(actual) ^ set(expected)
    assert not diff

    expected = _features_transformers['zscore']
    actual = get_transformer('zscore', target=False)

    assert isinstance(actual, expected)

    expected = _features_transformers['zscore']
    actual = get_transformer('zscore', target=True)

    assert isinstance(actual, TargetTransfromerWrapper)
    assert isinstance(actual.transformer, expected)

    with pytest.raises(ValueError, match="is not available"):
        get_transformer('scaler_robust', target=True)

    with pytest.raises(ValueError, match="is not available"):
        get_transformer('wrong', target=False)
