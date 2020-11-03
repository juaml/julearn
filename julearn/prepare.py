from julearn.transformers.target import TargetTransfromerWrapper
import pandas as pd
import numpy as np
from copy import deepcopy
from sklearn.model_selection import RepeatedKFold, GridSearchCV
from sklearn.base import clone
from sklearn.model_selection import check_cv

from . estimators import get_model
from . transformers import get_transformer
from . scoring import get_extended_scorer
from . model_selection import wrap_search


def _validate_input_data(X, y, confounds, df, groups):
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


def prepare_input_data(X, y, confounds, df, pos_labels, groups):
    _validate_input_data(X, y, confounds, df, groups)
    # Declare them as None to avoid CI issues
    df_X_conf = None
    confound_names = None
    df_groups = None
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

        if groups is not None:
            if isinstance(groups, np.ndarray):
                columns = [f'group_{i}' for i in range(groups.shape[0])]
                df_groups = pd.DataFrame(groups, columns=columns)

    else:
        X_conf_columns = deepcopy(X) if isinstance(X, list) else [X]
        if isinstance(confounds, list):
            X_conf_columns.extend(confounds)
        elif isinstance(confounds, str):
            X_conf_columns.append(confounds)

        df_X_conf = df.loc[:, X_conf_columns].copy()
        y = df.loc[:, y].copy()
        if groups is not None:
            df_groups = df.loc[:, groups].copy()
        confound_names = confounds

    if pos_labels is not None:
        if not isinstance(pos_labels, list):
            pos_labels = [pos_labels]
        y = y.isin(pos_labels).astype(np.int)

    return df_X_conf, y, df_groups, confound_names


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
        model_name = model
        model = get_model(model_name, problem_type)
    elif _is_valid_sklearn_model(model):
        model_name = model.__class__.__name__.lower()
    else:
        raise ValueError(f'Model must be a string or a scikit-learn compatible'
                         ' object.')
    return model_name, model


def prepare_model_selection(msel_dict, pipeline, model_name, cv_outer):
    hyperparameters = msel_dict.get('hyperparameters', None)
    if hyperparameters is None:
        raise ValueError("The 'hyperparameters' value must be specified for "
                         "model selection.")
    cv_inner = msel_dict.get('cv', None)
    gs_scoring = msel_dict.get('scoring', None)

    if cv_inner is None:
        cv_inner = deepcopy(cv_outer)
    hyper_params = _prepare_hyperparams(hyperparameters, pipeline, model_name)

    if len(hyper_params) > 0:
        pipeline = wrap_search(
            GridSearchCV, pipeline, hyper_params, cv=cv_inner,
            scoring=gs_scoring)
    return pipeline


def _prepare_hyperparams(hyperparams, pipeline, model_name):

    def rename_param(param):
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

    to_tune = {}
    for param, val in hyperparams.items():
        # If we have more than 1 value, we will tune it. If not, it will
        # be set in the model.
        if hasattr(val, '__iter__') and not isinstance(val, str):
            if len(val) > 1:
                to_tune[rename_param(param)] = val
            else:
                pipeline.set_param(val)
        else:
            pipeline.set_params(**{rename_param(param): val})
    return to_tune


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

    preprocess_X = [_create_preprocess_tuple(transformer)
                    for transformer in preprocess_X]
    return preprocess_X


def _get_confound_transformer(conf):
    returned_features = 'unknown_same_type'
    if isinstance(conf, str):
        conf, returned_features = get_transformer(conf)
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
            preprocess_y = get_transformer(preprocess_y, target=True)
        elif not isinstance(preprocess_y, TargetTransfromerWrapper):
            if _is_valid_sklearn_transformer(preprocess_y):
                preprocess_y = TargetTransfromerWrapper(preprocess_y)
            else:
                raise ValueError(f'y preprocess must be a string or a '
                                 'valid sklearn transformer instance')
    return preprocess_y


def prepare_cv(cv):
    """Generates an CV using string compatible with
    repeat:5_nfolds:5 where 5 can be exchange with any int.
    Alternatively, it can take in a valid cv splitter or int as
    in cross_validate in sklearn.

    Parameters
    ----------
    cv : int or str or cv_splitter
        [description]

    """

    def parser(cv_string):
        n_repeats, n_folds = cv_string.split('_')
        n_repeats, n_folds = [int(name.split(':')[-1])
                              for name in [n_repeats, n_folds]]
        return RepeatedKFold(n_splits=n_folds, n_repeats=n_repeats)

    try:
        _cv = check_cv(cv)
    except ValueError:
        _cv = parser(cv)

    return _cv


def prepare_scoring(estimator, score_name):
    return get_extended_scorer(estimator, score_name)


def _create_preprocess_tuple(transformer):
    if type(transformer) == list:
        return transformer
    elif type(transformer) == str:
        trans_name = transformer
        trans, returned_features = get_transformer(transformer)
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
