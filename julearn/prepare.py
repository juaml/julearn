# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL
from julearn.model_selection.available_searchers import (get_searcher,
                                                         list_searchers)
from julearn.utils.column_types import pick_columns
from julearn.transformers.target import TargetTransfromerWrapper
import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.model_selection import RepeatedKFold
from sklearn.base import clone
from sklearn.model_selection import check_cv

from . estimators import get_model
from . transformers import get_transformer
from . scoring import get_extended_scorer
from . utils import raise_error, warn, logger
from . model_selection import (StratifiedGroupsKFold,
                               RepeatedStratifiedGroupsKFold)


def _validate_input_data_np(X, y, confounds, groups):
    # X must be np.ndarray with at most 2d
    if not isinstance(X, np.ndarray):
        raise_error(
            'X must be a numpy array if no dataframe is specified')

    if X.ndim not in [1, 2]:
        raise_error('X must be at most bi-dimensional')

    # Y must be np.ndarray with 1 dimension
    if not isinstance(y, np.ndarray):
        raise_error(
            'y must be a numpy array if no dataframe is specified')

    if y.ndim != 1:
        raise_error('y must be one-dimensional')

    # Same number of elements
    if X.shape[0] != y.shape[0]:
        raise_error(
            'The number of samples in X do not match y '
            '(X.shape[0] != y.shape[0]')

    if confounds is not None:
        if not isinstance(confounds, np.ndarray):
            raise_error(
                'confounds must be a numpy array if no dataframe is '
                'specified')

        if confounds.ndim not in [1, 2]:
            raise_error('confounds must be at most bi-dimensional')

        if X.shape[0] != confounds.shape[0]:
            raise_error(
                'The number of samples in X do not match confounds '
                '(X.shape[0] != confounds.shape[0]')

    if groups is not None:
        if not isinstance(groups, np.ndarray):
            raise_error(
                'groups must be a numpy array if no dataframe is '
                'specified')

        if groups.ndim != 1:
            raise_error('groups must be one-dimensional')


def _validate_input_data_df(X, y, confounds, df, groups):

    # in the dataframe
    if not isinstance(X, (str, list)):
        raise_error('X must be a string or list of strings')

    if not isinstance(y, str):
        raise_error('y must be a string')

    # Confounds can be a string, list or none
    if not isinstance(confounds, (str, list, type(None))):
        raise_error('If not None, confounds must be a string or list '
                    'of strings')

    if not isinstance(groups, (str, type(None))):
        raise_error('groups must be a string')

    if not isinstance(df, pd.DataFrame):
        raise_error('df must be a pandas.DataFrame')

    if any(not isinstance(x, str) for x in df.columns):
        raise_error('DataFrame columns must be strings')


def _validate_input_data_df_ext(X, y, confounds, df, groups):
    missing_columns = [t_x for t_x in X if t_x not in df.columns]
    # In reality, this is not needed as the regexp match will fail
    # Leaving it as additional check in case the regexp match changes
    if len(missing_columns) > 0:  # pragma: no cover
        raise_error(  # pragma: no cover
            'All elements of X must be in the dataframe. '
            f'The following are missing: {missing_columns}')

    if y not in df.columns:
        raise_error(
            f"Target '{y}' (y) is not a valid column in the dataframe")

    if confounds is not None:
        missing_columns = [
            t_c for t_c in confounds if t_c not in df.columns]
        # In reality, this is not needed as the regexp match will fail
        # Leaving it as additional check in case the regexp match changes
        if len(missing_columns) > 0:  # pragma: no cover
            raise_error(  # pragma: no cover
                'All elements of confounds must be in the dataframe. '
                f'The following are missing: {missing_columns}')

    if groups is not None:
        if groups not in df.columns:
            raise_error(f"Groups '{groups}' is not a valid column "
                        "in the dataframe")
        if groups == y:
            warn("y and groups are the same column")
        if groups in X:
            warn("groups is part of X")

    if y in X:
        warn(f'List of features (X) contains the target {y}')


def prepare_input_data(X, y, confounds, df, pos_labels, groups):
    """ Prepare the input data and variables for the pipeline

    Parameters
    ----------
    X : str, list(str) or numpy.array
        The features to use.
        See https://juaml.github.io/julearn/input.html for details.
    y : str or numpy.array
        The targets to predict.
        See https://juaml.github.io/julearn/input.html for details.
    confounds : str, list(str) or numpy.array | None
        The confounds.
        See https://juaml.github.io/julearn/input.html for details.
    df : pandas.DataFrame with the data. | None
        See https://juaml.github.io/julearn/input.html for details.
    pos_labels : str, int, float or list | None
        The labels to interpret as positive. If not None, every element from y
        will be converted to 1 if is equal or in pos_labels and to 0 if not.
    groups : str or numpy.array | None
        The grouping labels in case a Group CV is used.
        See https://juaml.github.io/julearn/input.html for details.

    Returns
    -------
    df_X_conf : pandas.DataFrame
        A dataframe with the features and confounds (if specified in the
        confounds parameter) for each sample.
    df_y : pandas.Series
        A series with the y variable (target) for each sample.
    df_groups : pandas.Series
        A series with the grouping variable for each sample (if specified
        in the groups parameter).
    confound_names : str
        The name of the columns if df_X_conf that represent confounds.

    """
    logger.info('==== Input Data ====')

    # Declare them as None to avoid CI issues
    df_X_conf = None
    confound_names = None
    df_groups = None
    if df is None:
        logger.info(f'Using numpy arrays as input')
        _validate_input_data_np(X, y, confounds, groups)
        # creating df_X_conf
        if X.ndim == 1:
            X = X[:, None]
        logger.info(f'# Samples: {X.shape[0]}')
        logger.info(f'# Features: {X.shape[1]}')
        columns = [f'feature_{i}' for i in range(X.shape[1])]
        df_X_conf = pd.DataFrame(X, columns=columns)

        # adding confounds to df_X_conf
        if confounds is not None:
            if confounds.ndim == 1:
                confounds = confounds[:, None]
            logger.info(f'# Confounds: {X.shape[1]}')
            confound_names = [
                f'confound_{i}' for i in range(confounds.shape[1])]
            df_X_conf[confound_names] = confounds

        # creating a Series for y if not existent
        df_y = pd.Series(y, name='y')

        if groups is not None:
            logger.info('Using groups')
            df_groups = pd.Series(groups, name='groups')

    else:
        logger.info(f'Using dataframe as input')
        _validate_input_data_df(X, y, confounds, df, groups)
        logger.info(f'Features: {X}')
        logger.info(f'Target: {y}')
        if confounds is not None:
            logger.info(f'Confounds: {confounds}')
            X_confounds = pick_columns(confounds, df.columns)
        else:
            X_confounds = []

        if X == [':']:
            X_columns = [x for x in df.columns if x not in X_confounds
                         and x != y]
            if groups is not None:
                X_columns = [x for x in X_columns if x not in groups]
        else:
            X_columns = pick_columns(X, df.columns)

        logger.info(f'Expanded X: {X_columns}')
        logger.info(f'Expanded Confounds: {X_confounds}')
        _validate_input_data_df_ext(X_columns, y, X_confounds, df, groups)
        X_conf_columns = X_columns

        overlapping = [t_c for t_c in X_confounds if t_c in X]
        if len(overlapping) > 0:
            warn(f'X contains the following confounds {overlapping}')
        for t_c in X_confounds:
            # This will add the confounds if not there already
            if t_c not in X_conf_columns:
                X_conf_columns.append(t_c)

        df_X_conf = df.loc[:, X_conf_columns].copy()
        df_y = df.loc[:, y].copy()
        if groups is not None:
            logger.info(f'Using {groups} as groups')
            df_groups = df.loc[:, groups].copy()
        confound_names = X_confounds

    if pos_labels is not None:
        if not isinstance(pos_labels, list):
            pos_labels = [pos_labels]
        logger.info(f'Setting the following as positive labels {pos_labels}')
        # TODO: Warn if pos_labels are not in df_y
        df_y = df_y.isin(pos_labels).astype(np.int64)
    logger.info('====================')
    logger.info('')
    return df_X_conf, df_y, df_groups, confound_names


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
    model_name : str
        The model name
    model : object
        The model

    """
    logger.info('====== Model ======')
    if isinstance(model, str):
        logger.info(f'Obtaining model by name: {model}')
        model_name = model
        model = get_model(model_name, problem_type)
    elif _is_valid_sklearn_model(model):
        model_name = model.__class__.__name__.lower()
        logger.info(f'Using scikit-learn model: {model_name}')
    else:
        raise_error(
            f'Model must be a string or a scikit-learn compatible object.')
    logger.info('===================')
    logger.info('')
    return model_name, model


def prepare_model_params(msel_dict, pipeline):
    """Prepare model parameters.

    For each of the model parameters, determine if it can be directly set or
    must be tuned using hyperparameter tuning.

    Parameters
    ----------
    msel_dict : dict
        A dictionary with the model selection parameters.The dictionary can
        define the following keys:

        * 'STEP__PARAMETER': A value (or several) to be used as PARAMETER for
          STEP in the pipeline. Example: 'svm__probability': True will set
          the parameter 'probability' of the 'svm' model. If more than option
        * 'search': The kind of search algorithm to use e.g.:
          'grid' or 'random'. All valid julearn searchers can be entered.
        * 'cv': If search is going to be used, the cross-validation
          splitting strategy to use. Defaults to same CV as for the model
          evaluation.
        * 'scoring': If search is going to be used, the scoring metric to
          evaluate the performance.
        * 'search_params': Additional parameters for the search method.

    pipeline : ExtendedDataframePipeline
        The pipeline to apply/tune the hyperparameters

    Returns
    -------
    pipeline : ExtendedDataframePipeline
        The modified pipeline
    """
    logger.info('= Model Parameters =')

    tunning_params = ['scoring', 'cv', 'search', 'search_params']

    hyperparameters = {k: v for k, v in msel_dict.items()
                       if k not in tunning_params}

    hyper_params = _prepare_hyperparams(hyperparameters, pipeline)

    if len(hyper_params) > 0:
        scoring = msel_dict.get('scoring', None)
        search = msel_dict.get('search', 'grid')
        search_params = msel_dict.get('search_params', {})
        cv_inner = search_params.get('cv', None)
        cv_inner_dep = msel_dict.get('cv', None)
        if cv_inner_dep is not None:
            warn(
                "`cv` should not be directly provided in the"
                "`model_params` anymore. This functionality will"
                "be removed in the next version of julearn."
                "Please use `cv` inside of `search_params` instead",
                category=DeprecationWarning


            )
        cv_inner = cv_inner_dep if cv_inner is None else cv_inner

        if search in list_searchers():
            logger.info(f'Tunning hyperparameters using {search}')
            search = get_searcher(search)
        else:
            if isinstance(search, str):
                raise_error(
                    f'The searcher {search} is not a valid julearn searcher. '
                    'You can get a list of all available once by using: '
                    'julearn.model_selection.list_searchers(). You can also '
                    'enter a valid scikit-learn searcher or register it.'
                )
            else:
                warn(
                    f'{search} is not a registered searcher. '
                )
                logger.info(
                    f'Tunning hyperparameters using not registered {search}')

        logger.info('Hyperparameters:')
        for k, v in hyper_params.items():
            logger.info(f'\t{k}: {v}')

        cv_inner = prepare_cv(cv_inner)

        search_params['cv'] = cv_inner
        search_params['scoring'] = scoring
        logger.info('Search Parameters:')
        for k, v in search_params.items():
            logger.info(f'\t{k}: {v}')
        pipeline = search(pipeline, hyper_params, **search_params)
    else:
        if 'cv' in msel_dict:
            warn('Hyperparameter search CV was specified, but no '
                 'hyperparameters to tune')
        if 'scoring' in msel_dict:
            warn('Hyperparameter search scoring was specified, but no '
                 'hyperparameters to tune')
        if 'search' in msel_dict:
            warn('Hyperparameter search method was specified, but no '
                 'hyperparameters to tune')
    logger.info('====================')
    logger.info('')
    return pipeline


def _prepare_hyperparams(hyperparams, pipeline):
    """Prepare model hyperparameters.

    Either set the model hyperparameter or add it to the dictionary of
    parameters to be tuned.

    Parameters
    ----------
    hyperparams : dict
        A dictionary with hyperparameters. The key must be like
        'STEP__PARAMETER': A value (or several) to be used as PARAMETER for
        STEP in the pipeline.
    pipeline : ExtendedDataframePipeline
        The pipeline to apply the hyperparameters

    Returns
    -------
    to_tune : dict
        The parameters that must be tuned.
    """
    to_tune = {}
    # steps = list(pipeline.named_steps.keys())
    for param, val in hyperparams.items():
        # If we have more than 1 value, we will tune it. If not, it will
        # be set in the model.
        if hasattr(val, '__iter__') and not isinstance(val, str):
            if len(val) > 1:
                to_tune[param] = val
            else:
                logger.info(f'Setting hyperparameter {param} = {val[0]}')
                pipeline.set_params(**{param: val[0]})
        else:
            logger.info(f'Setting hyperparameter {param} = {val}')
            pipeline.set_params(**{param: val})
    return to_tune


def prepare_preprocessing(preprocess_X, preprocess_y, preprocess_conf,
                          confounds):
    if preprocess_X is not None and not isinstance(preprocess_X, list):
        preprocess_X = [preprocess_X]

    preprocess_X = _prepare_preprocess_X(preprocess_X, confounds)
    preprocess_y = _prepare_preprocess_y(preprocess_y)
    if preprocess_conf is not None:
        preprocess_conf = _prepare_preprocess_confounds(preprocess_conf)
    return preprocess_X, preprocess_y, preprocess_conf


def _prepare_preprocess_X(preprocess_X, confounds):
    '''
    validates preprocess_X and returns a list of tuples accordingly
    and default params for this list
    '''
    preprocess_X = None if preprocess_X == [] else preprocess_X
    if preprocess_X is None:
        if confounds is not None:
            preprocess_X = [_create_preprocess_tuple('remove_confound')]
    else:
        preprocess_X = [_create_preprocess_tuple(transformer)
                        for transformer in preprocess_X]
        names, _, = zip(*preprocess_X)
        if ('remove_confound' not in names) and (confounds is not None):
            preprocess_X = ([(_create_preprocess_tuple('remove_confound'))]
                            + preprocess_X)
    return preprocess_X


def _prepare_preprocess_confounds(preprocess_conf):
    '''
    uses user input to create a list of tuples for a normal pipeline
    this can then be used for transforming the confounds/z
    '''
    if preprocess_conf is not None:
        if not isinstance(preprocess_conf, list):
            preprocess_conf = [preprocess_conf]
        elif preprocess_conf == []:
            preprocess_conf = None

    if preprocess_conf is not None:
        preprocess_conf = [
            _create_preprocess_tuple(transformer)
            for transformer in preprocess_conf
        ]

    return preprocess_conf


def _prepare_preprocess_y(preprocess_y):
    if preprocess_y is not None:
        if isinstance(preprocess_y, str):
            preprocess_y = get_transformer(preprocess_y, target=True)
        elif not isinstance(preprocess_y, TargetTransfromerWrapper):
            if _is_valid_sklearn_transformer(preprocess_y):
                preprocess_y = TargetTransfromerWrapper(preprocess_y)
            else:
                raise_error(f'y preprocess must be a string or a '
                            'valid sklearn transformer instance')
    return preprocess_y


def prepare_cv(cv):
    """Generates a CV using string compatible with
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
        logger.info(f'CV interpreted as RepeatedKFold with {n_repeats} '
                    f'repetitions of {n_folds} folds')
        return RepeatedKFold(n_splits=n_folds, n_repeats=n_repeats)

    try:
        _cv = check_cv(cv)
        logger.info(f'Using scikit-learn CV scheme {_cv}')
    except ValueError:
        _cv = parser(cv)

    return _cv


def prepare_scoring(estimator, scorers):
    """Prepares the scikit-learn scorers to work with the
    ExtendedDataFramePipeline

    Parameters
    ----------
    estimator : julearn.pipeline.ExtendedDataFramePipeline
        An estimator with a .transform_confounds and .transform_target
        method needed for scoring against a new ground truth.
    scorers : str, obj, list(str) or dict
        A scorer name (or list of) or dict of scorer name:scorer.
        For more information see:
        https://scikit-learn.org/stable/modules/model_evaluation.html#scoring



    Returns
    -------
    scoring : scorer | dict(string, scorer)
        A dictionary with the corresponding scorers for each scorer name
        or scorer,
        suitable for sklearn.model_selection.cross_validate.
    """
    if scorers is None:
        return None
    if isinstance(scorers, list):
        scoring = {k: get_extended_scorer(estimator, k) for k in scorers}
    elif isinstance(scorers, dict):
        scoring = {
            name: get_extended_scorer(estimator, scorer) if isinstance(
                scorer, str) else scorer
            for name, scorer in scorers.items()}
    else:
        scoring = get_extended_scorer(estimator, scorers)
    return scoring


def _create_preprocess_tuple(transformer):
    if type(transformer) == list:
        return transformer
    elif type(transformer) == str:
        trans_name = transformer
        trans = get_transformer(transformer)
    else:
        trans_name = transformer.__class__.__name__.lower()
        trans = clone(transformer)

    return trans_name, trans


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


def check_consistency(
        pipeline, preprocess_X, preprocess_y, preprocess_confounds, df_X_conf,
        y, cv, groups, problem_type):
    """Check the consistency of the parameters/input"""

    # Check problem type and the target.
    n_classes = np.unique(y.values).shape[0]
    if problem_type == 'binary_classification':
        # If not exactly two classes:
        if n_classes != 2:
            if preprocess_y is None:
                raise_error(
                    f'The number of classes ({n_classes}) is not suitable for '
                    'a binary classification. You can either specify '
                    '``pos_labels``, a suitable y transformer or change the '
                    'problem type.')
            else:
                warn(
                    f'The number of classes ({n_classes}) is not suitable for '
                    'a binary classification. However, a y transformer has '
                    'been set.')
    elif problem_type == 'multiclass_classification':
        if n_classes == 2:
            warn(
                f'A multiclass classification will be performed but only 2 '
                'classes are defined in y.')
    else:
        # Regression
        is_numeric = np.issubdtype(y.values.dtype, np.number)
        if not is_numeric:
            if preprocess_y is None:
                raise_error(
                    f'The kind of values in y ({y.values.dtype}) is not '
                    'suitable for a regression. You can either specify a '
                    'suitable y transformer or change the problem type.')
            else:
                warn(
                    f'The kind of values in y ({y.values.dtype}) is not '
                    'suitable for a regression. However, a y transformer has '
                    'been set.')
        else:
            n_classes = np.unique(y.values).shape[0]
            if n_classes == 2:
                warn(
                    f'A regression will be performed but only 2 '
                    'distinct values are defined in y.')
    # Check groups and CV scheme
    if groups is not None:
        valid_instances = (
            model_selection.GroupKFold,
            model_selection.GroupShuffleSplit,
            model_selection.LeaveOneGroupOut,
            model_selection.LeavePGroupsOut,
            StratifiedGroupsKFold,
            RepeatedStratifiedGroupsKFold
        )
        if not isinstance(cv, valid_instances):
            warn('The parameter groups was specified but the CV strategy '
                 'will not consider them.')
