# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL
from . dataframe import DropColumns, ChangeColumnTypes
from .. utils import raise_error, warn
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

"""
a dictionary containing all supported transformers
name : [sklearn transformer,
        str:returns_same_features]
"""

_available_transformers = {
    # Decomposition
    'pca': [PCA, 'unknown'],
    # Scalers
    'zscore': [StandardScaler, 'same'],
    'scaler_robust': [RobustScaler, 'same'],
    'scaler_minmax': [MinMaxScaler, 'same'],
    'scaler_maxabs': [MaxAbsScaler, 'same'],
    'scaler_normalizer': [Normalizer, 'same'],
    'scaler_quantile': [QuantileTransformer, 'same'],
    'scaler_power': [PowerTransformer, 'same'],
    # Feature selection
    'select_univariate': [GenericUnivariateSelect, 'subset'],
    'select_percentile': [SelectPercentile, 'subset'],
    'select_k': [SelectKBest, 'subset'],
    'select_fdr': [SelectFdr, 'subset'],
    'select_fpr': [SelectFpr, 'subset'],
    'select_fwe': [SelectFwe, 'subset'],
    'select_variance': [VarianceThreshold, 'subset'],
    # DataFrame operations
    'remove_confound': [
        DataFrameConfoundRemover,
        'from_transformer'
    ],
    'drop_columns': [DropColumns, 'subset'],
    'change_column_types': [ChangeColumnTypes, 'from_transformer']
}

_available_transformers_reset = deepcopy(_available_transformers)
_apply_to_default_exceptions = {
    'remove_confound': ['continuous', 'confound'],
    'drop_columns': 'all',
    'change_column_types': 'all'
}

_available_target_transformers = {
    'zscore': StandardScaler,
    'remove_confound': [TargetConfoundRemover, 'same'],
}

_dict_transformer_to_name = {transformer: name
                             for name, (transformer, apply_to) in deepcopy(
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


def _get_returned_features(transformer):
    transformer_name = _dict_transformer_to_name.get(transformer.__class__)
    returned_features = _available_transformers.get(transformer_name)[1]
    if returned_features is None:
        warn(f'The transformer {transformer_name} is not a registered '
             'transformer. '
             'Therefore, `returned_features` will be set to `unknown`.'
             'In other words variable names cannot be preserved after this '
             'transformer. If you want to change this use '
             '`julearn.transformer.register_transformer` to register your'
             'transformer'
             )
        return 'unknown'
    else:
        return returned_features


def _get_apply_to(transformer):
    transformer_name = _dict_transformer_to_name.get(transformer.__class__)
    if isinstance(transformer_name, str):

        if (transformer_name.startswith('select')) and (
                transformer_name in list_transformers()):
            apply_to = 'all_features'

        else:
            apply_to = _apply_to_default_exceptions.get(transformer_name,
                                                        'continuous')
    else:
        apply_to = 'continuous'

    return apply_to


def register_transformer(transformer_name, transformer,
                         returned_features, apply_to):

    if _available_transformers.get(transformer_name) is not None:
        warn(f'The transformer_name `{transformer_name}` does already exist. '
             'Therefore, you are overwriting this transformer.'
             )
    _available_transformers[transformer_name] = [transformer,
                                                 _get_returned_features]

    _dict_transformer_to_name[transformer.__class__] = transformer_name


def reset_register():
    global _available_transformers
    _available_transformers = deepcopy(
        _available_transformers_reset)
    return _available_transformers
