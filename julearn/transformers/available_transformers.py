# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL
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
    'zscore': [StandardScaler(), 'same'],
    'pca': [PCA(), 'unknown'],
    'remove_confound': [
        DataFrameConfoundRemover(),
        'subset',
    ],
    'passthrough': [PassThroughTransformer(), 'same']
}


_available_target_transformers = {
    'zscore': TargetTransfromerWrapper(StandardScaler()),
    'passthrough': TargetPassThroughTransformer()

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
