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
from . cbpm import CBPM
from . dataframe import DropColumns, ChangeColumnTypes, FilterColumns
from .. utils import raise_error, warn, logger
from . confounds import DataFrameConfoundRemover

"""
a dictionary containing all supported transformers
name : [sklearn transformer,
        str:returns_same_features]
"""

_available_transformers = {
    # Decomposition
    'pca': PCA,
    # Scalers
    'zscore': StandardScaler,
    'scaler_robust': RobustScaler,
    'scaler_minmax': MinMaxScaler,
    'scaler_maxabs': MaxAbsScaler,
    'scaler_normalizer': Normalizer,
    'scaler_quantile': QuantileTransformer,
    'scaler_power': PowerTransformer,
    # Feature selection
    'select_univariate': GenericUnivariateSelect,
    'select_percentile': SelectPercentile,
    'select_k': SelectKBest,
    'select_fdr': SelectFdr,
    'select_fpr': SelectFpr,
    'select_fwe': SelectFwe,
    'select_variance': VarianceThreshold,
    # Custom
    'cbpm': CBPM,
    # DataFrame operations
    'remove_confound': DataFrameConfoundRemover,
    'drop_columns': DropColumns,
    'change_column_types': ChangeColumnTypes,
    'filter_columns': FilterColumns,
}

_available_transformers_reset = deepcopy(_available_transformers)

_dict_transformer_to_name = {
    transformer: name
    for name, transformer in _available_transformers.items()
}


def list_transformers():
    """List all the available transformers

    Returns
    -------
    out : list(str)
        A list will all the available transformer names.
    """
    return list(_available_transformers.keys())


def get_transformer(name, **params):
    """Get a transformer

    Parameters
    ----------
    name : str
        The transformer name

    Returns
    -------
    out : scikit-learn compatible transformer
        The transformer object.
    """
    out = None
    if name not in _available_transformers:
        raise_error(
            f'The specified transformer ({name}) is not available. '
            f'Valid options are: {list(_available_transformers.keys())}')
    trans = _available_transformers[name]
    out = trans(**params)
    return out


def register_transformer(transformer_name, transformer_cls, overwrite=None):
    """Register a transformer to julearn.
    This function allows you to add a transformer to julearn.
    Afterwards, it behaves like every other julearn transformer and can
    be referred to by name. E.g. you can use its name in `preprocess_X`

    Parameters
    ----------
    transformer_name : str
        Name by which the transformer will be referenced by
    transformer_cls : object
        The class by which the transformer can be initialized from.
    overwrite : bool | None, optional
        decides whether overwrite should be allowed, by default None.
        Options are:

        * None : overwrite is possible, but warns the user
        * True : overwrite is possible without any warning
        * False : overwrite is not possible, error is raised instead
    """
    if _available_transformers.get(transformer_name) is not None:

        if overwrite is None:
            warn(
                f'Transformer named {transformer_name}'
                ' already exists. '
                f'Therefore, {transformer_name} will be overwritten. '
                'To remove this warning set overwrite=True. '
                'If you want to reset this use '
                '`julearn.transformer.reset_models`.'
            )
        elif overwrite is False:
            raise_error(
                f'Transformer named {transformer_name}'
                ' already exists. '
                f'Therefore, {transformer_name} will be overwritten. '
                'overwrite is set to False, '
                'therefore you cannot overwrite '
                'existing models. Set overwrite=True'
                ' in case you want to '
                'overwrite existing transformer.'
            )

    logger.info(f'registering transformer named {transformer_name}.'
                )

    _dict_transformer_to_name[transformer_cls] = transformer_name
    _available_transformers[transformer_name] = transformer_cls


def reset_transformer_register():
    global _available_transformers
    global _dict_transformer_to_name
    _available_transformers = deepcopy(_available_transformers_reset)

    _dict_transformer_to_name = {
        transformer: name
        for name, transformer in _available_transformers.items()
    }
    return _available_transformers
