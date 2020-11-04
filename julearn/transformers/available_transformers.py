from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone

from . confounds import DataFrameConfoundRemover
from . basic import PassThroughTransformer
from . target import TargetTransfromerWrapper, TargetPassThroughTransformer
from .. utils import raise_error

"""
a dictionary containing all supported transformers
name : [sklearn transformer,
        str:returns_same_features]
"""

_available_transformers = {
    'z_score': [StandardScaler(), 'same'],
    'pca': [PCA(), 'unknown'],
    'remove_confound': [
        DataFrameConfoundRemover(),
        'subset',
    ],
    'passthrough': [PassThroughTransformer(), 'same']
}


_available_target_transformers = {
    'z_score': TargetTransfromerWrapper(StandardScaler()),
    'passthrough': TargetPassThroughTransformer()

}


def list_transformers(target=False):
    out = None
    if target is False:
        out = list(_available_transformers.keys())
    else:
        out = list(_available_target_transformers.keys())
    return out


def get_transformer(name, target=False):
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
