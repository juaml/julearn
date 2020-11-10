# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL
from sklearn.decomposition import PCA
from sklearn.preprocessing import (StandardScaler, RobustScaler, MaxAbsScaler,
                                   MinMaxScaler, Normalizer,
                                   QuantileTransformer, PowerTransformer)
from sklearn.feature_selection import (GenericUnivariateSelect,
                                       SelectPercentile, SelectKBest,
                                       SelectFdr, SelectFpr, SelectFwe,
                                       VarianceThreshold)
from sklearn.base import clone

from . confounds import DataFrameConfoundRemover
from . target import TargetTransfromerWrapper
from .. utils import raise_error

"""
a dictionary containing all supported transformers
name : [sklearn transformer,
        str:returns_same_features]
"""

_available_transformers = {
    'pca': [PCA(), 'unknown'],
    'remove_confound': [
        DataFrameConfoundRemover(),
        'subset',
    ],
    # Scalers
    'zscore': [StandardScaler(), 'same'],
    'scaler_robust': [RobustScaler(), 'same'],
    'scaler_minmax': [MinMaxScaler(), 'same'],
    'scaler_maxabs': [MaxAbsScaler(), 'same'],
    'scaler_normalizer': [Normalizer(), 'same'],
    'scaler_quantile': [QuantileTransformer(), 'same'],
    'scaler_power': [PowerTransformer(), 'same'],
    # Feature selection
    'select_univariate': [GenericUnivariateSelect(), 'subset'],
    'select_percentile': [SelectPercentile(), 'subset'],
    'select_k': [SelectKBest(), 'subset'],
    'select_fdr': [SelectFdr(), 'subset'],
    'select_fpr': [SelectFpr(), 'subset'],
    'select_fwe': [SelectFwe(), 'subset'],
    'select_variance': [VarianceThreshold(), 'subset']
}


_available_target_transformers = {
    'zscore': TargetTransfromerWrapper(StandardScaler()),
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


def get_transformer(name, target=False):
    """Get a transfomer

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
        trans, same = _available_transformers[name]
        out = clone(trans), same
    else:
        if name not in _available_target_transformers:
            raise_error(
                f'The specified target transformer ({name}) is not available. '
                f'Valid options are: '
                f'{list(_available_target_transformers.keys())}')
        out = clone(_available_target_transformers[name])
    return out
