"""Provide registry of transformers."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL

from copy import deepcopy
from typing import Any, List

from sklearn.decomposition import PCA
from sklearn.feature_selection import (
    GenericUnivariateSelect,
    SelectFdr,
    SelectFpr,
    SelectFwe,
    SelectKBest,
    SelectPercentile,
    VarianceThreshold,
)
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)

from ..utils import logger, raise_error, warn_with_log
from ..utils.typing import TransformerLike
from .cbpm import CBPM
from .confound_remover import ConfoundRemover
from .dataframe import ChangeColumnTypes, DropColumns, FilterColumns


_available_transformers = {
    # Scalers
    "zscore": StandardScaler,
    "scaler_robust": RobustScaler,
    "scaler_minmax": MinMaxScaler,
    "scaler_maxabs": MaxAbsScaler,
    "scaler_normalizer": Normalizer,
    "scaler_quantile": QuantileTransformer,
    "scaler_power": PowerTransformer,
    # Feature selection
    "select_univariate": GenericUnivariateSelect,
    "select_percentile": SelectPercentile,
    "select_k": SelectKBest,
    "select_fdr": SelectFdr,
    "select_fpr": SelectFpr,
    "select_fwe": SelectFwe,
    "select_variance": VarianceThreshold,
    # DataFrame operations
    "confound_removal": ConfoundRemover,
    "drop_columns": DropColumns,
    "change_column_types": ChangeColumnTypes,
    "filter_columns": FilterColumns,
    # Decomposition
    "pca": PCA,
    # Custom
    "cbpm": CBPM,
}

_available_transformers_reset = deepcopy(_available_transformers)


def list_transformers() -> List[str]:
    """List all the available transformers.

    Returns
    -------
    list of str
        A list will all the available transformer names.

    """
    return list(_available_transformers.keys())


def get_transformer(name: str, **params: Any) -> TransformerLike:
    """Get a transformer.

    Parameters
    ----------
    name : str
        The transformer name.
    **params : dict
        Parameters to get transformer.

    Returns
    -------
    scikit-learn compatible transformer
        The transformer object.

    """
    out = None
    if name not in _available_transformers:
        raise_error(
            f"The specified transformer ({name}) is not available. "
            f"Valid options are: {list(_available_transformers.keys())}"
        )
    trans = _available_transformers[name]
    out = trans(**params)
    return out


def register_transformer(transformer_name, transformer_cls, overwrite=None):
    """Register a transformer to julearn.

    This function allows you to add a transformer to julearn.
    Afterwards, it behaves like every other julearn transformer and can
    be referred to by name.

    Parameters
    ----------
    transformer_name : str
        Name by which the transformer will be referenced by
    transformer_cls : object
        The class by which the transformer can be initialized from.
    overwrite : bool, optional
        Whether overwrite should be allowed. Options are:

        * None : overwrite is possible, but warns the user (default).
        * True : overwrite is possible without any warning.
        * False : overwrite is not possible, error is raised instead.

    Raises
    ------
    ValueError
        If `transformer_name` is already registered and `overwrite` is False.

    Warns
    -----
    RuntimeWarning
        If `transformer_name` is already registered and `overwrite` is None.
    """
    if _available_transformers.get(transformer_name) is not None:
        if overwrite is None:
            warn_with_log(
                f"Transformer named {transformer_name} already exists. "
                f"Therefore, {transformer_name} will be overwritten. To "
                "remove this warning set overwrite=True."
            )
        elif overwrite is False:
            raise_error(
                f"Transformer named {transformer_name} already exists "
                "and overwrite is set to False. Set `overwrite=True` "
                "in case you want to overwrite an existing transformer."
            )

    logger.info(f"registering transformer named {transformer_name}.")

    _available_transformers[transformer_name] = transformer_cls


def reset_transformer_register():
    """Reset the transformer register to its initial state."""

    global _available_transformers
    _available_transformers = deepcopy(_available_transformers_reset)

    return _available_transformers
