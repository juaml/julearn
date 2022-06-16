# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL
from . cbpm import CBPM
from . dataframe import DropColumns, ChangeColumnTypes
from .. utils import raise_error, warn, logger
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
from . target import TargetTransfromerWrapper, is_targettransformer

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
    # Custom
    'cbpm': [CBPM, 'unknown'],
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
        trans = _available_target_transformers[name]
        out = trans(**params)
        if not is_targettransformer(out):
            out = TargetTransfromerWrapper(out)
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
                         returned_features, apply_to,
                         overwrite=None):
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

            * 'same': leads copies the names from the original pd.DataFrame
            * 'subset': leads to the columns being a subset of the original
              pd.DataFrame. This functionality needs the transformer to have a
              .get_support method following sklearn standards.
            * 'from_transformer': the outputted columns are already defined in
              the transformer
            * 'unknown' leads to created column names,
            * 'unknown_same_type' leads to created column names
              with the same column type.
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

    if apply_to != 'continuous':
        _apply_to_default_exceptions[transformer_name] = apply_to

    _available_transformers[transformer_name] = [
        transformer_cls, returned_features]


def reset_transformer_register():
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
