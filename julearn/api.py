"""API for julearn."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL

import hashlib
import inspect
import json

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import check_cv, cross_validate
from sklearn.model_selection._split import (
    PredefinedSplit,
    _CVIterableWrapper,
)
from sklearn.pipeline import Pipeline

from .pipeline import PipelineCreator
from .prepare import check_consistency, prepare_input_data
from .scoring import check_scoring
from .utils import logger, raise_error


def _recurse_to_list(a):
    """Recursively convert a to a list."""
    if isinstance(a, (list, tuple)):
        return [_recurse_to_list(i) for i in a]
    elif isinstance(a, np.ndarray):
        return a.tolist()
    else:
        return a


def _compute_cvmdsum(cv):
    """Compute the sum of the CV generator."""
    params = {k: v for k, v in vars(cv).items()}
    params["class"] = cv.__class__.__name__

    out = None

    # Check for special cases that might not be comparable
    if "random_state" in params:
        if params["random_state"] is None:
            if params.get("shuffle", True) is True:
                # If it's shuffled and the random seed is None
                out = "non-reproducible"

    if isinstance(cv, _CVIterableWrapper):
        splits = params.pop("cv")
        params["cv"] = _recurse_to_list(splits)
    if isinstance(cv, PredefinedSplit):
        params["test_fold"] = params["test_fold"].tolist()
        params["unique_folds"] = params["unique_folds"].tolist()

    if "cv" in params:
        if inspect.isclass(params["cv"]):
            params["cv"] = params["cv"].__class__.__name__

    if out is None:
        out = hashlib.md5(
            json.dumps(params, sort_keys=True).encode("utf-8")
        ).hexdigest()

    return out


def run_cross_validation(
    X,
    y,
    model,
    X_types=None,
    data=None,
    problem_type=None,
    preprocess=None,
    return_estimator=False,
    return_train_score=False,
    cv=None,
    groups=None,
    scoring=None,
    pos_labels=None,
    model_params=None,
    search_params=None,
    seed=None,
    n_jobs=None,
    verbose=0,
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
    X_types : dict[str, list of str]
        A dictionary containing keys with column type as a str and the
        columns of this column type as a list of str.
    data : pandas.DataFrame | None
        DataFrame with the data (optional).
        See https://juaml.github.io/julearn/input.html for details.
    problem_type : str
        The kind of problem to model.

        Options are:

        * "classification": Perform a classification
          in which the target (y) has categorical classes (default).
          The parameter pos_labels can be used to convert a target with
          multiple_classes into binary.
        * "regression". Perform a regression. The target (y) has to be
          ordinal at least.

    preprocess : str, TransformerLike or list or PipelineCreator | None
        Transformer to apply to the features. If string, use one of the
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

        * None: defaults to 5-fold
        * int: the number of folds in a `(Stratified)KFold`
        * CV Splitter (see scikit-learn documentation on CV)
        * An iterable yielding (train, test) splits as arrays of indices.

    groups : str or numpy.array | None
        The grouping labels in case a Group CV is used.
        See https://juaml.github.io/julearn/input.html for details.
    scoring : ScorerLike, optional
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

    search_params : dict | None
        Additional parameters in case Hyperparameter Tuning is performed, with
        the following keys:

        * 'kind': The kind of search algorithm to use, e.g.:
            'grid' or 'random'. Can be any valid julearn searcher name or
            scikit-learn compatible searcher.
        * 'cv': If a searcher is going to be used, the cross-validation
            splitting strategy to use. Defaults to same CV as for the model
            evaluation.
        * 'scoring': If a searcher is going to be used, the scoring metric to
            evaluate the performance.

        See https://juaml.github.io/julearn/hyperparameters.html for details.
    seed : int | None
        If not None, set the random seed before any operation. Useful for
        reproducibility.
    verbose: int
        Verbosity level of outer cross-validation.
        Follows scikit-learn/joblib converntions.
        0 means no additional information is printed.
        Larger number genereally mean more information is printed.
        Note: verbosity up to 50 will print into standard error,
        while larger than 50 will print in standrad output.

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

    """

    X_types = {} if X_types is None else X_types
    if seed is not None:
        # If a seed is passed, use it, otherwise do not do anything. User
        # might have set the seed outside of the library
        logger.info(f"Setting random seed to {seed}")
        np.random.seed(seed)

    # Interpret the input data and prepare it to be used with the library
    df_X, y, df_groups, X_types = prepare_input_data(
        X=X,
        y=y,
        df=data,
        pos_labels=pos_labels,
        groups=groups,
        X_types=X_types,
    )

    if model_params is None:
        model_params = {}

    if search_params is None:
        search_params = {}

    # Deal with model and preprocess
    if isinstance(model, Pipeline):
        raise_error(
            "Currently we do not allow to pass a scikit-learn pipeline as "
            "model, use PipelineCreator instead"
        )

    if isinstance(model, PipelineCreator):
        if preprocess is not None:
            raise_error(
                "If model is a PipelineCreator, preprocess should be None"
            )
        if problem_type is not None:
            raise_error("Problem type should be set in the PipelineCreator")

        if len(model_params) > 0:
            raise_error(
                "If model is a PipelineCreator, model_params must be None. "
                f"Currently, it contains {model_params.keys()}"
            )

        pipeline = model.to_pipeline(
            X_types=X_types, search_params=search_params
        )
        problem_type = model.problem_type

    elif not isinstance(model, (str, BaseEstimator)):
        raise_error(
            "Model has to be a PipelineCreator, a string or a "
            "scikit-learn compatible model."
        )
    else:
        if problem_type is None:
            raise_error(
                "If model is not a PipelineCreator, then `problem_type` "
                "must be specified in run_cross_validation."
            )
        if isinstance(preprocess, str):
            preprocess = [preprocess]
        if isinstance(preprocess, list):
            pipeline_creator = PipelineCreator.from_list(
                preprocess, model_params, problem_type=problem_type
            )
        elif preprocess is None:
            pipeline_creator = PipelineCreator(problem_type=problem_type)
        else:
            raise_error(
                "preprocess has to be a string or a " "list of strings."
            )

        # Add the model to the pipeline creator
        t_params = {}
        if isinstance(model, str):
            t_params = {
                x.replace(f"{model}__", ""): y
                for x, y in model_params.items()
                if x.startswith(f"{model}__")
            }
        else:
            model_name = model.__class__.__name__.lower()
            if any(x.startswith(f"{model_name}__") for x in model_params):
                raise_error(
                    "Cannot use model_params with a model object. Use either "
                    "a string or a PipelineCreator"
                )
        pipeline_creator.add(step=model, **t_params)

        # Check for extra model_params that are not used
        unused_params = []
        for t_param in model_params:
            used = False
            for step in pipeline_creator.steps:
                if t_param.startswith(f"{step.name}__"):
                    used = True
                    break
            if not used:
                unused_params.append(t_param)
        if len(unused_params) > 0:
            raise_error(
                "The following model_params are incorrect: " f"{unused_params}"
            )
        pipeline = pipeline_creator.to_pipeline(
            X_types=X_types, search_params=search_params
        )

    # Log some information
    logger.info("= Data Information =")
    logger.info(f"\tProblem type: {problem_type}")
    logger.info(f"\tNumber of samples: {len(df_X)}")
    logger.info(f"\tNumber of features: {len(df_X.columns)}")
    logger.info("====================")
    logger.info("")

    if problem_type == "classification":
        logger.info(f"\tNumber of classes: {len(np.unique(y))}")
        logger.info(f"\tTarget type: {y.dtype}")
        logger.info(f"\tClass distributions: {y.value_counts()}")
    elif problem_type == "regression":
        logger.info(f"\tTarget type: {y.dtype}")

    # Prepare cross validation
    cv_outer = check_cv(cv)  # type: ignore
    logger.info(f"Using outer CV scheme {cv_outer}")

    check_consistency(y, cv, groups, problem_type)

    cv_return_estimator = return_estimator in ["cv", "all"]
    scoring = check_scoring(pipeline, scoring)

    cv_mdsum = _compute_cvmdsum(cv_outer)

    scores = cross_validate(
        pipeline,
        df_X,
        y,
        cv=cv_outer,
        scoring=scoring,
        groups=df_groups,
        return_estimator=cv_return_estimator,
        n_jobs=n_jobs,
        return_train_score=return_train_score,
        verbose=verbose,
    )

    n_repeats = getattr(cv_outer, "n_repeats", 1)
    n_folds = len(scores["fit_time"]) // n_repeats

    repeats = np.repeat(np.arange(n_repeats), n_folds)
    folds = np.tile(np.arange(n_folds), n_repeats)

    fold_sizes = np.array(
        [list(map(len, x)) for x in cv_outer.split(df_X, y, groups=df_groups)]
    )
    scores["n_train"] = fold_sizes[:, 0]
    scores["n_test"] = fold_sizes[:, 1]
    scores["repeat"] = repeats
    scores["fold"] = folds
    scores["cv_mdsum"] = cv_mdsum

    out = pd.DataFrame(scores)
    if return_estimator in ["final", "all"]:
        pipeline.fit(df_X, y)
        out = out, pipeline

    return out
