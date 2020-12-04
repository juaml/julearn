# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL
from copy import deepcopy
from sklearn.decomposition import PCA
from sklearn.preprocessing import (StandardScaler, RobustScaler, MaxAbsScaler,
                                   MinMaxScaler, Normalizer,
                                   QuantileTransformer, PowerTransformer)
from sklearn.feature_selection import (GenericUnivariateSelect,
                                       SelectPercentile, SelectKBest,
                                       SelectFdr, SelectFpr, SelectFwe,
                                       VarianceThreshold)
from . confounds import DataFrameConfoundRemover, TargetConfoundRemover
from . target import TargetTransfromerWrapper
from .. utils import raise_error
from .tmp_transformers import DropColumns, ChangeColumnTypes

"""
a dictionary containing all supported transformers
name : [sklearn transformer,
        str:returns_same_features]
"""

_available_transformers = {
    # Decomposition
    'pca': [PCA, 'unknown', 'continuous'],
    # Scalers
    'zscore': [StandardScaler, 'same', 'continuous'],
    'scaler_robust': [RobustScaler, 'same', 'continuous'],
    'scaler_minmax': [MinMaxScaler, 'same', 'continuous'],
    'scaler_maxabs': [MaxAbsScaler, 'same', 'continuous'],
    'scaler_normalizer': [Normalizer, 'same', 'continuous'],
    'scaler_quantile': [QuantileTransformer, 'same', 'continuous'],
    'scaler_power': [PowerTransformer, 'same', 'continuous'],
    # Feature selection
    'select_univariate': [GenericUnivariateSelect, 'subset', 'continuous'],
    'select_percentile': [SelectPercentile, 'subset', 'continuous'],
    'select_k': [SelectKBest, 'subset', 'continuous'],
    'select_fdr': [SelectFdr, 'subset', 'continuous'],
    'select_fpr': [SelectFpr, 'subset', 'continuous'],
    'select_fwe': [SelectFwe, 'subset', 'continuous'],
    'select_variance': [VarianceThreshold, 'subset', 'continuous'],
    # DataFrame operations
    'remove_confound': [
        DataFrameConfoundRemover,
        'from_transformer', ['continuous', 'confound']
    ],
    'drop_columns': [DropColumns, 'subset', 'all'],
    'change_column_types': [ChangeColumnTypes, 'from_transformer', 'all']
}


_available_target_transformers = {
    'zscore': StandardScaler,
    'remove_confound': [TargetConfoundRemover, 'same'],
}

_runtime_transformer_dict = {
    transformer: [returned_features, apply_to]
    for _, (transformer, returned_features, apply_to) in deepcopy(
        _available_transformers).items()
}


def list_transformers(target=False):
    """List all the available transformers

    Parameters
    ----------
    target : bool
        If True, return a list of the target tranformers. If False (default),
        return a list of features/confounds transformers.

    Returns
    -------
    out : list(str)
        A list will all the available transformer names.
    """
    out = None
    if target is False:
        out = list(_available_transformers.keys())
    else:
        out = list(_available_target_transformers.keys())
    return out


def get_transformer(name, target=False, **params):
    """Get a transformer

    Parameters
    ----------
    name : str
        The transformer name
    target : bool
        If True, return a target tranformer. If False (default),
        return a features/confounds transformers.

    Returns
    -------
    out : scikit-learn compatible transformer
        The transformer object.
    """
    out = None
    if target is False:
        if name not in _available_transformers:
            raise_error(
                f'The specified transformer ({name}) is not available. '
                f'Valid options are: {list(_available_transformers.keys())}')
        trans, *_ = _available_transformers[name]
        out = trans(**params)
    else:
        if name not in _available_target_transformers:
            raise_error(
                f'The specified target transformer ({name}) is not available. '
                f'Valid options are: '
                f'{list(_available_target_transformers.keys())}')
        trans = _available_target_transformers[name]()
        out = TargetTransfromerWrapper(trans)
    return out


def get_returned_features(transformer):
    return _runtime_transformer_dict.get(transformer.__class__)[0]


def get_apply_to(transformer):
    return _runtime_transformer_dict.get(transformer.__class__)[1]


def register_transformer(transformer,
                         returned_features='same', apply_to='continuous'):
    _runtime_transformer_dict[transformer.__class__] = [
        returned_features, apply_to]
