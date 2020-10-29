from julearn.transformers.target import TargetTransfromerWrapper
import pandas as pd
import numpy as np
from copy import deepcopy
from sklearn.model_selection import RepeatedKFold
from sklearn.base import clone
from sklearn.model_selection import check_cv

from . estimators import available_models
from . transformers import (available_transformers,
                            available_target_transformers)
from . metrics import get_extended_scorer


def _validate_input_data(X, y, confounds, df):

    if df is None:
        # Case 1: we don't have a dataframe in df

        # X is either an ndarray or dataframe
        assert isinstance(X, (np.ndarray, pd.DataFrame))

        # X is bidimentional
        assert X.ndim == 2

        # Same for y, but only one dimension
        assert isinstance(y, (np.ndarray, pd.Series))
        assert y.ndim == 1

        # Same number of elements
        assert X.shape[0] == y.shape[0]

        if confounds is not None:
            # Confounds is any kind of array or pandas object
            assert isinstance(confounds, (np.ndarray, pd.DataFrame, pd.Series))
            assert confounds.ndim in [1, 2]
            assert confounds.shape[0] == y.shape[0]

    else:
        # Case 2: we have a dataframe. X, y and confounds must be columns
        # in the dataframe
        assert isinstance(X, (str, list))
        assert isinstance(y, str)

        # Confounds can be a string, list or none
        assert isinstance(confounds, (str, list, type(None)))

        assert isinstance(df, pd.DataFrame)

        if isinstance(X, list):
            assert all(Xi in df.columns for Xi in X)
        else:
            assert X in df.columns

        assert y in df.columns

        if isinstance(confounds, str):
            assert confounds in df.columns
        elif isinstance(confounds, list):
            assert all(x in df.columns for x in confounds)


def prepare_input_data(X, y, confounds, df, pos_labels):
    _validate_input_data(X, y, confounds, df)
    # Declare them as None to avoid CI issues
    df_X_conf = None
    confound_names = None
    if df is None:
        # creating df_X_conf
        if isinstance(X, np.ndarray):
            columns = [f'feature_{i}' for i in range(X.shape[0])]
            df_X_conf = pd.DataFrame(X, columns=columns)
        elif isinstance(X, pd.DataFrame):
            df_X_conf = X.copy()

        # adding confounds to df_X_conf
        if isinstance(confounds, (pd.Series, pd.DataFrame)):
            # TODO: Merge the dataframe, keep the index. Use join/merge.
            # Raise an error if it can't be merged.
            if isinstance(confounds, pd.Series):
                confound_names = confounds.name
            else:
                confound_names = confounds.columns
            df_X_conf[confound_names] = confounds

        elif isinstance(confounds, np.ndarray):
            confound_names = [
                f'confound_{i}' for i in range(confounds.shape[0])]
            df_X_conf[confound_names] = confounds

        # creating a Series for y if not existent
        if isinstance(y, np.ndarray):
            y = pd.Series(y, name='y')

    else:
        X_conf_columns = deepcopy(X) if isinstance(X, list) else [X]
        if isinstance(confounds, list):
            X_conf_columns.extend(confounds)
        elif isinstance(confounds, str):
            X_conf_columns.append(confounds)

        df_X_conf = df.loc[:, X_conf_columns].copy()
        y = df.loc[:, y].copy()
        confound_names = confounds

    if pos_labels is not None:
        if not isinstance(pos_labels, list):
            pos_labels = [pos_labels]
        y = y.isin(pos_labels).astype(np.int)

    return df_X_conf, y, confound_names


def prepare_model(model, problem_type):
    """ Get the propel model/name pair from the input

    Parameters
    ----------
    model: str or sklearn.base.BaseEstimator
        str/model_name that can be read in to create a model.
    problem_type: str
        binary_classification, multiclass_classification or regression

    Returns
    -------
    out : tuple(str, model)
        A tuple with the model name and object.

    """
    if isinstance(model, str):
        if model not in available_models:
            available_model_names = list(available_models.keys())
            raise ValueError(
                f'The specified model ({model}) is not available. '
                f'Valid options are: {available_model_names}')

        model_name = model
        model = available_models[model][problem_type]

    elif _is_valid_sklearn_model(model):
        model_name = model.__class__.__name__.lower()
    else:
        raise ValueError(f'Model must be a string or a scikit-learn compatible'
                         ' object.')
    return model_name, model


def prepare_hyperparams(hyperparams, model_name):

    def recode_param(param):
        first, *rest = param.split('__')

        if first == 'features':
            new_first = 'dataframe_pipeline'
        elif first == 'confounds':
            new_first = 'confound_dataframe_pipeline'
        elif first == 'target':
            new_first = 'y_transformer'
        elif first == model_name:
            new_first = 'dataframe_pipeline__' + first

        else:
            raise ValueError(
                'Each element of the hyperparameters dict  has to start with '
                f'"features__", "confounds__", "target__" or "{model_name}__" '
                f'but was {first}')
        return '__'.join([new_first] + rest)

    return {recode_param(param): val for param, val in hyperparams.items()}


def prepare_preprocessing(preprocess_X, preprocess_y, preprocess_confounds):

    preprocess_X = _prepare_preprocess_X(preprocess_X)
    preprocess_y = _prepare_preprocess_y(preprocess_y)
    preprocess_conf = _prepare_preprocess_confounds(preprocess_confounds)
    return preprocess_X, preprocess_y, preprocess_conf


def _prepare_preprocess_X(preprocess_X):
    '''
    validates preprocess_X and returns a list of tuples accordingly
    and default params for this list
    '''

    for step in preprocess_X:
        if type(step) == str:
            if available_transformers.get(step) is None:
                raise ValueError(f'{step} in preprocess_X is not'
                                 'an available transformer')
        else:
            if _is_valid_sklearn_transformer(step) is False:
                raise ValueError(f'{step} in preprocess has to be either'
                                 'a string or a valid sklearn transformer'

                                 )

    preprocess_X = [_create_preprocess_tuple(transformer)
                    for transformer in preprocess_X]
    return preprocess_X


def _get_confound_transformer(conf):
    returned_features = 'unknown_same_type'
    if isinstance(conf, str):
        if conf in available_transformers:
            conf, returned_features = available_transformers[conf]
        else:
            _valid_names = list(available_transformers.keys())
            raise ValueError(
                f'The specified confound preprocessing ({conf}) is '
                f'not available. Valid options are: {_valid_names}')
    elif not _is_valid_sklearn_transformer(conf):
        raise ValueError(
            f'The specified confound preprocessing ({conf}) is not valid.'
            f'It has to be a string or sklearn transformer.')
    return conf, returned_features


def _prepare_preprocess_confounds(preprocess_conf):
    '''
    uses user input to create a list of tuples for a normal pipeline
    this can then be used for transforming the confounds/z
    '''
    if not isinstance(preprocess_conf, list):
        preprocess_conf = [preprocess_conf]

    returned_features = 'unknown_same_type'
    for step in preprocess_conf:
        _, returned_features = _get_confound_transformer(step)

        if returned_features == 'unknown':
            returned_features = 'unknown_same_type'

    preprocess_conf = [
        # returned_features ignored
        _create_preprocess_tuple(transformer)
        for transformer in preprocess_conf

    ]
    # replace returned_feature with what we got here
    preprocess_conf = [step[:2] + (returned_features,) + (step[-1],)
                       for step in preprocess_conf]

    return preprocess_conf


def _prepare_preprocess_y(preprocess_y):
    if preprocess_y is not None:
        if isinstance(preprocess_y, str):
            if preprocess_y not in available_target_transformers:
                _valid_names = list(available_target_transformers.keys())
                raise ValueError(
                    f'The specified target preprocessing ({preprocess_y}) is '
                    f'not available. Valid options are: {_valid_names}')
            preprocess_y = available_target_transformers[preprocess_y]
        elif not isinstance(preprocess_y, TargetTransfromerWrapper):
            if _is_valid_sklearn_transformer(preprocess_y):
                preprocess_y = TargetTransfromerWrapper(preprocess_y)
            else:
                raise ValueError(f'y preprocess must be a string or a '
                                 'valid sklearn transformer instance')
    return preprocess_y


def prepare_cv(cv_outer, cv_inner):
    """Generates an outer and inner cv using string compatible with
    repeat:5_nfolds:5 where 5 can be exchange with any int.
    Alternatively, it can take in a valid cv splitter or int as
    in cross_validate in sklearn.

    Parameters
    ----------
    cv_outer : int or str or cv_splitter
        [description]
    cv_inner : int or str or cv_splitter
        [description]
    """

    def parser(cv_string):
        n_repeats, n_folds = cv_string.split('_')
        n_repeats, n_folds = [int(name.split(':')[-1])
                              for name in [n_repeats, n_folds]]
        return RepeatedKFold(n_splits=n_folds, n_repeats=n_repeats)

    def convert_to_cv(cv):
        try:
            _cv = check_cv(cv)
        except ValueError:
            _cv = parser(cv)

        return _cv

    cv_outer = convert_to_cv(cv_outer)
    if cv_inner == 'same':
        cv_inner = deepcopy(cv_outer)
    else:
        cv_inner = convert_to_cv(cv_inner)

    return cv_outer, cv_inner


def prepare_scoring(estimator, score_name):
    return get_extended_scorer(estimator, score_name)


def _create_preprocess_tuple(transformer):
    if type(transformer) == list:
        return transformer
    elif type(transformer) == str:
        trans_name = transformer
        trans, returned_features = available_transformers[transformer]
    else:
        trans_name = transformer.__class__.__name__.lower()
        trans = clone(transformer)
        returned_features = 'unknown'

    transform_columns = (['continuous', 'confound']
                         if trans_name == 'remove_confound'
                         else 'continuous')

    return trans_name, trans, returned_features, transform_columns


def _is_valid_sklearn_transformer(transformer):

    return (hasattr(transformer, 'fit') and
            hasattr(transformer, 'transform') and
            hasattr(transformer, 'get_params') and
            hasattr(transformer, 'set_params'))


def _is_valid_sklearn_model(model):
    return (hasattr(model, 'fit') and
            hasattr(model, 'predict') and
            hasattr(model, 'get_params') and
            hasattr(model, 'set_params'))
