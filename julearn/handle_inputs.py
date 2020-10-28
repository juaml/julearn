import pandas as pd
import numpy as np
from copy import deepcopy
from sklearn.model_selection import RepeatedKFold
from .available_estimators import (
    available_models,
    available_transformers,
    available_target_transformers)

from sklearn.base import clone
from sklearn.model_selection import check_cv


def validate_data_input(X, y, confounds, df):

    if df is None:
        assert type(X) == np.ndarray or type(X) == pd.DataFrame
        assert len(X.shape) == 2

        assert type(y) == np.ndarray or type(y) == pd.Series
        assert len(y.shape) == 1

        assert (
            type(confounds) == np.ndarray
            or type(confounds) == pd.DataFrame
            or type(confounds) == pd.Series
        )
        assert len(confounds.shape) == 1 or len(confounds.shape) == 2

    else:
        assert type(X) == str or type(X) == list
        assert type(y) == str
        assert type(confounds) == str or type(
            confounds) == list or confounds is None
        assert type(df) == pd.DataFrame

        if type(X) == list:
            for Xi in X:
                assert Xi in df.columns
        else:
            assert X in df.columns
        assert y in df.columns
        if type(confounds) == str:
            assert confounds in df.columns


def read_data_input(X, y, confounds, df):

    if df is None:
        # creating df_X_conf
        if type(X) == np.ndarray:
            X_columns = [f"feature_{i}" for i in range(len(X[0]))]
            df_X_conf = pd.DataFrame(X, columns=X_columns)
        elif type(X) == pd.DataFrame:
            df_X_conf = X.copy()

        # adding confounds to df_X_conf
        if type(confounds) == pd.Series:
            confound_names = confounds.name
            df_X_conf[confound_names] = confounds

        elif type(confounds) == pd.DataFrame:
            confound_names = confounds.columns
            df_X_conf[confound_names] = confounds

        elif type(confounds) == np.ndarray:
            if len(confounds.shape) == 1:
                confound_names = "confound"
                df_X_conf[confound_names] = confounds
            else:
                confound_names = [
                    f"confound_{i}" for i in range(len(confounds[0]))]
                df_X_conf[confound_names] = confounds

        # creating a Series for y if not existent
        if type(y) == np.ndarray:
            y = pd.Series(y, name="y")

    else:
        X_conf_columns = deepcopy(X) if type(X) == list else [deepcopy(X)]
        X_conf_columns.extend(confounds) if type(
            confounds
        ) == list else X_conf_columns.append(confounds)

        df_X_conf = df.loc[:, X_conf_columns].copy()
        y = df.loc[:, y].copy()
        confound_names = confounds

    return df_X_conf, y, confound_names


def read_validate_model(model, problem_type):
    """read in a model string and problem type or sklearn model

    Parameters
    ----------

    model: str or sklearn.base.BaseEstimator
        str/model_name that can be read in to create a model.
    problem_type: str
        binary_classification, multiclass_classification or regression

    Returns
    -------
    tuple(str, model), dict
        A tuple contasining a string name and model and
        additionally a dict containing the model_params

    """
    if type(model) == str:
        if available_models.get(model) is not None:
            model_name = model
            model = available_models[model][problem_type]
        else:
            raise ValueError('model is a string.'
                             'models provided as strings have to be'
                             'an available model'

                             )
    elif _is_valid_sklearn_transformer(model):
        model_name = model.__class__.__name__.lower()
    else:
        raise ValueError(f'model= {model} is neither a valid string'
                         'nor a valid sklearn mode')
    return model_name, model


def read_validate_user_params(params, model_name):

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
            raise ValueError('Paramsdict has to start with: '
                             f'features__, confounds__, target__,'
                             f' {model_name}__ for each params,but was {first}')
        return '__'.join([new_first] + rest)

    return {recode_param(param): val
            for param, val in params.items()}


def read_validate_preprocessing(preprocess_X,
                                preprocess_y,
                                preprocess_confounds):

    preprocess_X = read_validate_preprocess_X(preprocess_X)
    preprocess_y = read_validate_preprocess_y(preprocess_y)
    preprocess_conf = read_validate_preprocess_conf(preprocess_confounds)
    return preprocess_X, preprocess_y, preprocess_conf


def read_validate_preprocess_X(preprocess_X):
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
                    for transformer in preprocess_X

                    ]
    return preprocess_X


def read_validate_preprocess_conf(preprocess_conf):
    '''
    uses user input to create a list of tuples for a normal pipeline
    this can then be used for transforming the confounds/z
    '''
    if type(preprocess_conf) == str:

        if available_transformers.get(preprocess_conf) is not None:
            preprocess_conf = [preprocess_conf]
        else:
            raise ValueError(f'preprocess_conf is {preprocess_conf}'
                             'when it is a string it has to be a valid'
                             'available_transformer'
                             )
    for step in preprocess_conf:
        if type(step) == str:
            if available_transformers.get(step) is None:
                raise ValueError(f'{step} in preprocess_conf is not'
                                 'an available transformer')
            else:
                _,  returned_features = available_transformers[step]

                if returned_features == 'unknown':
                    returned_features = 'unknown_same_type'
        else:
            returned_features = 'unknown_same_type'
            if _is_valid_sklearn_transformer(step) is False:
                raise ValueError(f'{step} in preprocess has to be either'
                                 'a string or a valid sklearn transformer'

                                 )
    preprocess_conf = [
        # returned_features ignored
        _create_preprocess_tuple(transformer)
        for transformer in preprocess_conf

    ]
    # replace returned_feature with what we got here
    preprocess_conf = [step[:2] + (returned_features,) + (step[-1],)
                       for step in preprocess_conf]

    return preprocess_conf


def read_validate_preprocess_y(preprocess_y):

    if available_target_transformers.get(preprocess_y) is None:
        raise ValueError(f'preprocess_y has to an available '
                         'target_transformer'
                         f', but is {preprocess_y}')
    preprocess_y = available_target_transformers[preprocess_y]
    return preprocess_y


def read_validate_cv(cv_outer, cv_inner):
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
