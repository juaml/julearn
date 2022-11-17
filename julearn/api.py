# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL
import numpy as np
from sklearn.model_selection import cross_validate
import pandas as pd

from . prepare import (prepare_input_data,
                       prepare_model,
                       prepare_cv,
                       prepare_model_params,
                       prepare_preprocessing,
                       prepare_scoring,
                       check_consistency)
from . pipeline import _create_extended_pipeline

from . utils import logger


def run_cross_validation(
    X, y, model,
    data=None,
    confounds=None,
    problem_type='binary_classification',
    preprocess_X=None,
    preprocess_y=None,
    preprocess_confounds=None,
    return_estimator=False,
    return_train_score=False,
    cv=None,
    groups=None,
    scoring=None,
    pos_labels=None,
    model_params=None,
    seed=None,
    n_jobs=None,
    verbose=0
):
    """Run cross validation and score.

    Parameters
    ----------
    X : str, list(str) or numpy.array
        The features to use.
        See https://juaml.github.io/julearn/input.html for details.
    y : str or numpy.array
        The targets to predict.
        See https://juaml.github.io/julearn/input.html for details.
    model : str or scikit-learn compatible model.
        If string, it will use one of the available models.
        See :mod:`.available_models`.
    data : pandas.DataFrame | None
        DataFrame with the data (optional).
        See https://juaml.github.io/julearn/input.html for details.
    confounds : str, list(str) or numpy.array | None
        The confounds.
        See https://juaml.github.io/julearn/input.html for details.
    problem_type : str
        The kind of problem to model.

        Options are:

        * "binary_classification": Perform a binary classification
          in which the target (y) has only two possible classes (default).
          The parameter pos_labels can be used to convert a target with
          multiple_classes into binary.
        * "multiclass_classification": Performs a multiclass classification
          in which the target (y) has more than two possible values.
        * "regression". Perform a regression. The target (y) has to be
          ordinal at least.

    preprocess_X : str, scikit-learn compatible transformers or list | None
        Transformer to apply to the features (X). If string, use one of the
        available transformers. If list, each element can be a string or
        scikit-learn compatible transformer. If None (default), no
        transformation is applied.

        See documentation for details.
    preprocess_y : str or scikit-learn transformer | None
        Transformer to apply to the target (y). If None (default), no
        transformation is applied.

        See documentation for details.
    preprocess_confounds : str, scikit-learn transformers or list | None
        Transformer to apply to the features (X). If string, use one of the
        available transformers. If list, each element can be a string or
        scikit-learn compatible transformer. If None (default), no
        transformation is applied.

        See documentation for details.
    return_estimator : str | None
        Return the fitted estimator(s).
        Options are:

        * 'final': Return the estimator fitted on all the data.
        * 'cv': Return the all the estimator from each CV split, fitted on the
          training data.
        * 'all': Return all the estimators (final and cv).


    return_train_score : bool
        Whether to return the training score with the test scores 
        (default is False).

    cv : int, str or cross-validation generator | None
        Cross-validation splitting strategy to use for model evaluation.

        Options are:

        * None: defaults to 5-fold, repeated 5 times.
        * int: the number of folds in a `(Stratified)KFold`
        * CV Splitter (see scikit-learn documentation on CV)
        * An iterable yielding (train, test) splits as arrays of indices.

    groups : str or numpy.array | None
        The grouping labels in case a Group CV is used.
        See https://juaml.github.io/julearn/input.html for details.
    scoring : str | list(str) | obj | dict | None
        The scoring metric to use.
        See https://scikit-learn.org/stable/modules/model_evaluation.html for
        a comprehensive list of options. If None, use the model's default
        scorer.
    pos_labels : str, int, float or list | None
        The labels to interpret as positive. If not None, every element from y
        will be converted to 1 if is equal or in pos_labels and to 0 if not.
    model_params : dict | None
        If not None, this dictionary specifies the model parameters to use

        The dictionary can define the following keys:

        * 'STEP__PARAMETER': A value (or several) to be used as PARAMETER for
          STEP in the pipeline. Example: 'svm__probability': True will set
          the parameter 'probability' of the 'svm' model. If more than option
          is provided for at least one hyperparameter, a search will be
          performed.
        * 'search': The kind of search algorithm to use, e.g.:
          'grid' or 'random'. Can be any valid julearn searcher name or
          scikit-learn compatible searcher.
        * 'cv': If search is going to be used, the cross-validation
          splitting strategy to use. Defaults to same CV as for the model
          evaluation.
        * 'scoring': If search is going to be used, the scoring metric to
          evaluate the performance.
        * 'search_params': Additional parameters for the search method.

        See https://juaml.github.io/julearn/hyperparameters.html for details.
    seed : int | None
        If not None, set the random seed before any operation. Useful for
        reproducibility.

    Returns
    -------
    scores : pd.DataFrame
        The resulting scores (one column for each score specified).
        Additionally, a 'fit_time' column will be added.
        And, if ``return_estimator='all'`` or
        ``return_estimator='cv'``, an 'estimator' columns with the
        corresponding estimators fitted for each CV split.
    final_estimator : object
        The final estimator, fitted on all the data (only if
        ``return_estimator='all'`` or ``return_estimator='final'``)
    n_jobs : int | None
        Number of parallel jobs used by outer cross-validation.
        Follows scikit-learn/joblib conventions.
        None is 1 unless you use a joblib.parallel_backend.
        -1 means use all available processes for parallelisation.
    verbose: int
        Verbosity level of outer cross-validation.
        Follows scikit-learn/joblib converntions.
        0 means no additional information is printed.
        Larger number genereally mean more information is printed.
        Note: verbosity up to 50 will print into standard error,
        while larger than 50 will print in standrad output.
    """

    if seed is not None:
        # If a seed is passed, use it, otherwise do not do anything. User
        # might have set the seed outside of the library
        logger.info(f'Setting random seed to {seed}')
        np.random.seed(seed)

    if cv is None:
        logger.info(f'Using default CV')
        cv = 'repeat:5_nfolds:5'

    # Interpret the input data and prepare it to be used with the library
    df_X_conf, y, df_groups, _ = prepare_input_data(
        X=X, y=y, confounds=confounds, df=data, pos_labels=pos_labels,
        groups=groups)

    # create a the pipeline
    pipeline = create_pipeline(
        model=model,
        preprocess_X=preprocess_X,
        preprocess_y=preprocess_y,
        preprocess_confounds=preprocess_confounds,
        confounds=confounds,
        problem_type=problem_type,
        model_params=model_params
    )

    # Prepare cross validation
    cv_outer = prepare_cv(cv)

    scorer = prepare_scoring(pipeline, scoring)

    check_consistency(pipeline, preprocess_X, preprocess_y,
                      preprocess_confounds, df_X_conf, y, cv, groups,
                      problem_type)

    cv_return_estimator = return_estimator in ['cv', 'all']

    scores = cross_validate(pipeline, df_X_conf, y, cv=cv_outer,
                            scoring=scorer, groups=df_groups,
                            return_estimator=cv_return_estimator,
                            n_jobs=n_jobs,
                            return_train_score=return_train_score)

    n_repeats = getattr(cv_outer, 'n_repeats', 1)
    n_folds = len(scores['fit_time']) // n_repeats

    repeats = np.repeat(np.arange(n_repeats), n_folds)
    folds = np.tile(np.arange(n_folds), n_repeats)

    scores['repeat'] = repeats
    scores['fold'] = folds

    out = pd.DataFrame(scores)
    if return_estimator in ['final', 'all']:
        pipeline.fit(df_X_conf, y)
        out = out, pipeline

    return out


def create_pipeline(
    model,
    confounds=None,
    problem_type='binary_classification',
    preprocess_X=None,
    preprocess_y=None,
    preprocess_confounds=None,
    model_params=None
):
    """Creates a not fitted julearn pipeline.

    Parameters
    ----------
    model : str or scikit-learn compatible model.
        If string, it will use one of the available models.
        See :mod:`.available_models`.
    confounds : str, list(str) or numpy.array | None
        The confounds.
        See https://juaml.github.io/julearn/input.html for details.
    problem_type : str
        The kind of problem to model.

        Options are:

        * "binary_classification": Perform a binary classification
          in which the target (y) has only two possible classes (default).
          The parameter pos_labels can be used to convert a target with
          multiple_classes into binary.
        * "multiclass_classification": Performs a multiclass classification
          in which the target (y) has more than two possible values.
        * "regression". Perform a regression. The target (y) has to be
          ordinal at least.

    preprocess_X : str, scikit-learn compatible transformers or list | None
        Transformer to apply to the features (X). If string, use one of the
        available transformers. If list, each element can be a string or
        scikit-learn compatible transformer. If None (default), no
        transformation is applied.

        See documentation for details.
    preprocess_y : str or scikit-learn transformer | None
        Transformer to apply to the target (y). If None (default), no
        transformation is applied.

        See documentation for details.
    preprocess_confounds : str, scikit-learn transformers or list | None
        Transformer to apply to the features (X). If string, use one of the
        available transformers. If list, each element can be a string or
        scikit-learn compatible transformer. If None (default), no
        transformation is applied.

        See documentation for details.
    model_params : dict | None
        If not None, this dictionary specifies the model parameters to use

        The dictionary can define the following keys:

        * 'STEP__PARAMETER': A value (or several) to be used as PARAMETER for
          STEP in the pipeline. Example: 'svm__probability': True will set
          the parameter 'probability' of the 'svm' model. If more than option
          is provided for at least one hyperparameter, a search will be
          performed.
        * 'search': The kind of search algorithm to use, e.g.:
          'grid' or 'random'. Can be any valid julearn searcher name or
          scikit-learn compatible searcher.
        * 'cv': If search is going to be used, the cross-validation
          splitting strategy to use. Defaults to same CV as for the model
          evaluation.
        * 'scoring': If search is going to be used, the scoring metric to
          evaluate the performance.
        * 'search_params': Additional parameters for the search method.

        See https://juaml.github.io/julearn/hyperparameters.html for details.

    Returns
    -------
    pipeline : obj
        Not fitted julearn compatible pipeline
        or pipeline wrappen in Searcher.
    """

    # Interpret preprocessing parameters
    preprocess_vars = prepare_preprocessing(
        preprocess_X, preprocess_y, preprocess_confounds, confounds
    )
    preprocess_X, preprocess_y, preprocess_confounds = preprocess_vars
    # Prepare the model
    model_tuple = prepare_model(model=model, problem_type=problem_type)

    pipeline = _create_extended_pipeline(preprocess_X,
                                         preprocess_y,
                                         preprocess_confounds,
                                         model_tuple, confounds,
                                         categorical_features=None)

    if model_params is not None:
        pipeline = prepare_model_params(model_params, pipeline)

    return pipeline
