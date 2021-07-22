# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL
from copy import deepcopy
import numpy as np
import pandas as pd
from numpy import array_equal
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import (StandardScaler, RobustScaler, MaxAbsScaler,
                                   MinMaxScaler, Normalizer,
                                   QuantileTransformer, PowerTransformer)
from sklearn.feature_selection import (GenericUnivariateSelect,
                                       SelectPercentile, SelectKBest,
                                       SelectFdr, SelectFpr, SelectFwe,
                                       VarianceThreshold)

from . confounds import ConfoundRemover, TargetConfoundRemover
from . target import TargetTransformerWrapper, BaseTargetTransformer
from .. utils import raise_error, warn
from .. utils.array import safe_select

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
        ConfoundRemover,
        'from_transformer'
    ]
}

_available_transformers_reset = deepcopy(_available_transformers)
_apply_to_default_exceptions = {
    'remove_confound': ['continuous', 'confound'],
    'drop_columns': 'all',
}
_apply_to_default_exceptions_reset = deepcopy(_apply_to_default_exceptions)

_available_target_transformers = {
    'zscore': StandardScaler,
    'remove_confound': TargetConfoundRemover,
}

_dict_transformer_to_name = {transformer: name
                             for name, (transformer, apply_to) in (
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
    if target is True:
        avail = _available_target_transformers
    else:
        avail = _available_transformers

    if name not in avail:
        kind = 'target ' if target else ''
        raise_error(
            f'The specified {kind}transformer ({name}) is not available. '
            f'Valid options are: {list(avail.keys())}')
    trans, *_ = _available_transformers[name]
    out = trans(**params)
    if target is True and not isinstance(out, BaseTargetTransformer):
        out = TargetTransformerWrapper(out)

    return out


def _get_returned_features(transformer):
    transformer_name = _dict_transformer_to_name.get(
        transformer.__class__)
    if transformer_name is None:
        warn(f'The transformer {transformer} is not a registered '
             'transformer. '
             'Therefore, `returned_features` will be set to `unknown`.'
             'In other words variable names cannot be preserved after this '
             'transformer. If you want to change this use '
             '`julearn.transformer.register_transformer` to register your'
             'transformer'
             )
        returned_features = 'unknown'
    else:
        _, returned_features = _available_transformers.get(transformer_name)
    return returned_features


def _get_apply_to(transformer):
    transformer_name = (_dict_transformer_to_name.get(transformer.__class__))
    if isinstance(transformer_name, str):

        if (transformer_name.startswith('select')):
            apply_to = 'all_features'

        else:
            apply_to = _apply_to_default_exceptions.get(transformer_name,
                                                        'continuous')
    else:
        warn(f'The transformer {transformer} is not a registered '
             'transformer. '
             'Therefore, `apply_to` will be set to `continuous`.'
             'If you want to change this use '
             '`julearn.transformer.register_transformer` to register your'
             'transformer')
        apply_to = 'continuous'

    return apply_to


def register_transformer(transformer_name, transformer_cls,
                         returned_features, apply_to):
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
    returned_features : str
        Here, you can specify what features the transformer returns.
        The returned_features can be set to one of the following options:

            * 'same': The order and type of the columns of X are not
              modified.
            * 'subset': A subset of the columns of X are returned.
              This functionality needs the transformer to have a
              .get_support method following sklearn standards.
            * 'from_transformer': The resulting columns are already defined in
              the transformer
            * 'unknown': The resulting columns are unknonwn.
            * 'unknown_same_type' The resulting columns are unkown, but
              with the same type.

    apply_to : str | list(str)
        Defines to which columns the transformer is applied to.
        For this julearn user specified 'columns_types' from the user.
        All other columns will be ignored by the transformer and kept as
        they are.
        apply_to can be set to one or multiple of the following options:

            * 'all': The transformer is applied to all columns
            * 'all_features': The transformer is applied to continuous and
                categorical features.
            * 'continuous': The transformer is only applied to continuous
                features.
            * 'categorical': The transformer is only applied to categorical
                features.
            * 'confound': The transformer is only applied to confounds.

        As mentioned above you can combine these types.
        E.g. ['continuous', 'confound'] would specify that your transformer
        uses both the confounds and the continuous variables as input.
    """
    if _available_transformers.get(transformer_name) is not None:
        warn(f'The transformer of name `{transformer_name}` does already '
             'exist. Therefore, you are overwriting this transformer.'
             )
    _dict_transformer_to_name[transformer_cls] = transformer_name

    if apply_to != 'continuous':
        _apply_to_default_exceptions[transformer_name] = apply_to

    _available_transformers[transformer_name] = [
        transformer_cls, returned_features]


def reset_register():
    global _available_transformers
    global _dict_transformer_to_name
    global _apply_to_default_exceptions
    _available_transformers = deepcopy(
        _available_transformers_reset)

    _dict_transformer_to_name = {transformer: name
                                 for name, (transformer, apply_to) in (
                                     _available_transformers).items()
                                 }
    _apply_to_default_exceptions = deepcopy(_apply_to_default_exceptions_reset)
    return _available_transformers


def _propagate_transformer_column_names(transformer, X, column_names=None):
    if isinstance(X, np.ndarray):
        if column_names is None:
            raise_error(
                'You have to provide column_names when using np.arrays')
        else:
            column_names = np.array(column_names)
    elif isinstance(X, pd.DataFrame):
        column_names = np.array(
            X.columns) if column_names is None else np.array(column_names)
        if not array_equal(np.array(X.columns), column_names):
            raise_error(
                'Provided column_names and columns of the DataFrame '
                f'are not equal: DataFrame.columns={X.columns} '
                f'and column_names = {column_names}'
            )
    else:
        raise_error('X has to be either a pd.DataFrame or np.ndarray, '
                    'but X is of type {type(X)}'
                    )

    if isinstance(transformer, ColumnTransformer):
        # if there are more then 1 transformers we do not know how to
        # get proper column names, so we treat it as own unknown transformer
        if len(transformer.transformers) != 1:
            return _propagate_simple_transformer(transformer, X, column_names)

        _, inner_transformer, ind = transformer.transformers_[0]

        if isinstance(inner_transformer, ColumnTransformer):
            X_t = transformer.transform(X)
            trans_column_names = _propagate_transformer_column_names(
                inner_transformer, X_t, column_names[ind])
        else:
            trans_column_names = _propagate_simple_transformer(
                inner_transformer, safe_select(X, ind), column_names[ind])

        if transformer.remainder == 'passthrough':
            mask = np.zeros(column_names.shape, dtype=bool)
            mask[ind] = True
            trans_column_names = np.hstack(
                (trans_column_names, column_names[~mask]))

        return trans_column_names

    else:
        return _propagate_simple_transformer(transformer, X, column_names)


def _propagate_simple_transformer(transformer, X, column_names):
    return_type = _get_returned_features(transformer)
    if return_type == 'same':
        return column_names
    elif return_type == 'subset' or hasattr(transformer, 'subset'):
        mask = transformer.get_support()
        return column_names[mask]
    else:
        X_t = transformer.transform(X)
        transformer_name = _dict_transformer_to_name.copy().get(
            transformer.__class__)
        return [f'{transformer_name}_{i}' for i in range(len(X_t[0]))]
