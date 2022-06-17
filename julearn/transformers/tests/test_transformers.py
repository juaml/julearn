# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
# License: AGPL
from julearn.transformers.cbpm import CBPM
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
from sklearn.base import BaseEstimator, TransformerMixin
from seaborn import load_dataset

import pytest

from julearn.utils.testing import (do_scoring_test, PassThroughTransformer,
                                   TargetPassThroughTransformer)
from julearn.transformers import (
    list_transformers, get_transformer,
    reset_transformer_register, register_transformer,
    DataFrameConfoundRemover)
from julearn.transformers.available_transformers import (
    _get_returned_features, _get_apply_to,
    _available_transformers)


class fish(BaseEstimator, TransformerMixin):

    def __init__(self, can_it_fly):
        self.can_it_fly = can_it_fly

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X


reset_transformer_register()


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
    'cbpm': CBPM
}

_transformer_params = {
    'scaler_quantile': {'n_quantiles': 10},
    'select_k': {'k': 2},
}

_works_only_with_regression = [
    'cbpm'
]


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
        if tr_name in _works_only_with_regression:
            df_test = df_iris.copy()
            df_test[y] = df_iris[y].apply(
                lambda x: {'setosa': 0, 'versicolor': 1, 'virginica': 3}[x])
        else:
            df_test = df_iris.copy()
        do_scoring_test(X, y, data=df_test, api_params=api_params,
                        sklearn_model=clf, scorers=scorers)


def test_list_get_transformers():
    """Test list and getting transformers"""
    expected = list(_features_transformers.keys()) + [
        'pca',
        'remove_confound',
        'drop_columns',
        'change_column_types',
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


def test__get_returned_features():

    for name, transformer in _features_transformers.items():
        returned_features = _get_returned_features(transformer())
        assert returned_features == _available_transformers[name][1]

    with pytest.warns(RuntimeWarning, match=(
            'is not a registered '
            'transformer. '
            'Therefore, `returned_features`')
    ):
        returned_features = _get_returned_features(
            TargetPassThroughTransformer())

    assert returned_features == 'unknown'


def test__get_apply_to():
    apply_to_confound = _get_apply_to(DataFrameConfoundRemover())
    apply_to_select = _get_apply_to(get_transformer('select_percentile'))
    apply_to_zscore = _get_apply_to(StandardScaler())

    with pytest.warns(RuntimeWarning, match=(
            'is not a registered '
            'transformer. '
            'Therefore, `apply_to`')
    ):
        apply_to_pass = _get_apply_to(PassThroughTransformer())
    assert apply_to_zscore == apply_to_pass == 'continuous'
    assert apply_to_confound == ['continuous', 'confound']
    assert apply_to_select == 'all_features'


def test_register_reset():
    reset_transformer_register()
    with pytest.raises(ValueError, match='The specified transformer'):
        get_transformer('passthrough')

    register_transformer('passthrough', PassThroughTransformer,
                         'same', 'all')
    assert get_transformer('passthrough').__class__ == PassThroughTransformer
    assert _get_apply_to(PassThroughTransformer()) == 'all'
    assert _get_returned_features(PassThroughTransformer()) == 'same'

    with pytest.warns(RuntimeWarning, match='Transformer named'):
        register_transformer('passthrough', PassThroughTransformer,
                             'same', 'all')
    reset_transformer_register()
    with pytest.raises(ValueError, match='The specified transformer'):
        get_transformer('passthrough')

    register_transformer('passthrough', PassThroughTransformer,
                         'unknown', 'continuous')
    assert get_transformer('passthrough').__class__ == PassThroughTransformer
    assert _get_apply_to(PassThroughTransformer()) == 'continuous'
    assert _get_returned_features(PassThroughTransformer()) == 'unknown'


def test_register_class_no_default_params():

    reset_transformer_register()
    register_transformer('fish', fish, 'unknown', 'all')
    get_transformer('fish', can_it_fly='dont_be_stupid')


def test_get_target_transformer_no_error():
    get_transformer('zscore', target=True)
    get_transformer('remove_confound', target=True)


def test_register_warning():
    with pytest.warns(RuntimeWarning, match="Transformer name"):
        register_transformer('zscore', fish, 'unknown', 'all')
    reset_transformer_register()

    with pytest.raises(ValueError, match="Transformer name"):
        register_transformer('zscore', fish, 'unknown', 'all',
                             overwrite=False)
    reset_transformer_register()

    with pytest.warns(None) as record:
        register_transformer('zscore', fish, 'unknown', 'all',
                             overwrite=True)

    reset_transformer_register()
    assert len(record) == 0
