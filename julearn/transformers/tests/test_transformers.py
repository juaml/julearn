# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
# License: AGPL
from numpy.testing._private.utils import assert_array_equal
import pytest

from sklearn import svm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (StandardScaler, RobustScaler, MaxAbsScaler,
                                   MinMaxScaler, Normalizer,
                                   QuantileTransformer, PowerTransformer)
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer

from sklearn.feature_selection import (GenericUnivariateSelect,
                                       SelectPercentile, SelectKBest,
                                       SelectFdr, SelectFpr, SelectFwe,
                                       VarianceThreshold)
from sklearn.base import BaseEstimator, TransformerMixin
from seaborn import load_dataset


from julearn.utils.testing import (do_scoring_test, PassThroughTransformer,
                                   TargetPassThroughTransformer)
from julearn.transformers import (
    list_transformers, get_transformer, reset_register, register_transformer,
    ConfoundRemover)
from julearn.transformers.available_transformers import (
    _get_returned_features, _get_apply_to,
    _available_transformers,
    _propagate_simple_transformer, _propagate_transformer_column_names)

reset_register()


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
        'remove_confound'
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
    apply_to_confound = _get_apply_to(ConfoundRemover())
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
    reset_register()
    with pytest.raises(ValueError, match='The specified transformer'):
        get_transformer('passthrough')

    register_transformer('passthrough', PassThroughTransformer,
                         'same', 'all')
    assert get_transformer('passthrough').__class__ == PassThroughTransformer
    assert _get_apply_to(PassThroughTransformer()) == 'all'
    assert _get_returned_features(PassThroughTransformer()) == 'same'

    with pytest.warns(RuntimeWarning, match='The transformer of name '):
        register_transformer('passthrough', PassThroughTransformer,
                             'same', 'all')
    reset_register()
    with pytest.raises(ValueError, match='The specified transformer'):
        get_transformer('passthrough')

    register_transformer('passthrough', PassThroughTransformer,
                         'unknown', 'continuous')
    assert get_transformer('passthrough').__class__ == PassThroughTransformer
    assert _get_apply_to(PassThroughTransformer()) == 'continuous'
    assert _get_returned_features(PassThroughTransformer()) == 'unknown'


def test_register_class_no_default_params():

    class fish(BaseEstimator, TransformerMixin):

        def __init__(self, can_it_fly):
            self.can_it_fly = can_it_fly

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            return X

    reset_register()
    register_transformer('fish', fish, 'unknown', 'all')
    get_transformer('fish', can_it_fly='dont_be_stupid')


def test_get_target_transformer_no_error():
    get_transformer('zscore', target=True)
    get_transformer('remove_confound', target=True)


def test_propagation_of_columns_raises_errors():

    df_iris = load_dataset('iris')
    X = df_iris.iloc[:, :-1]
    new_columns = ['A', 'B', 'C', 'D']

    st = StandardScaler().fit(X)
    with pytest.raises(ValueError, match='Provided column_names and columns'):
        _propagate_transformer_column_names(st, X, new_columns)

    with pytest.raises(ValueError, match='You have to provide column_name'):
        _propagate_transformer_column_names(st, X.values)

    with pytest.raises(ValueError, match='X has to be either a pd.DataFrame'):
        _propagate_transformer_column_names(st, [1, 2, 3, 4])


def test_propagation_of_columns():
    df_iris = load_dataset('iris')
    X = df_iris.iloc[:, :-1]
    y = df_iris.species
    for t in [StandardScaler(), PCA(), SelectFdr()]:
        t = t.fit(X, y)
        df_propagate_columns = _propagate_transformer_column_names(
            t, X, X.columns)
        arr_propagate_columns = _propagate_transformer_column_names(
            t, X.values, X.columns)

        df_simp_propagate_columns = _propagate_simple_transformer(
            t, X, X.columns)

        arr_simp_propagate_columns = _propagate_simple_transformer(
            t, X.values, X.columns)

        assert_array_equal(df_propagate_columns, arr_propagate_columns)
        assert_array_equal(arr_propagate_columns, df_simp_propagate_columns)
        assert_array_equal(arr_propagate_columns, arr_simp_propagate_columns)


def test_ColumnTransformer_propagation_of_columns():
    # TODO test a selection transformer
    df_iris = load_dataset('iris')
    X = df_iris.iloc[:, :-1].values
    y = df_iris.species
    idx_slice = slice(2, 3)
    provided_columns = ['A', 'B', 'C', 'D']

    transformers = [
        ColumnTransformer([('st', StandardScaler(), idx_slice)],
                          remainder='passthrough'),
        ColumnTransformer([('st', StandardScaler(), idx_slice)],
                          remainder='drop'),
        ColumnTransformer([('pca', PCA(), idx_slice)],
                          remainder='passthrough'),
        ColumnTransformer([('pca', PCA(), idx_slice)],
                          remainder='drop'),
        ColumnTransformer([
            ('other', ColumnTransformer(
                [('st', StandardScaler(), slice(1, 2))], remainder='drop'),
             slice(1, None))], remainder='drop'
        )
    ]
    expected_columns_sets = [
        ['C', 'A', 'B', 'D'],
        ['C'],
        ['pca_0', 'A', 'B', 'D'],
        ['pca_0'],
        ['C']
    ]
    for t, expected_columns in zip(transformers, expected_columns_sets):
        t = t.fit(X, y)
        returned_columns = _propagate_transformer_column_names(
            t, X, provided_columns)

        assert_array_equal(returned_columns, expected_columns)
