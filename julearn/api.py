"""API for julearn."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL

from typing import List, Optional, Union, Dict


import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import check_cv, cross_validate
from sklearn.model_selection._search import BaseSearchCV
from sklearn.pipeline import Pipeline

from .pipeline import PipelineCreator
from .pipeline.merger import merge_pipelines
from .prepare import check_consistency, prepare_input_data
from .scoring import check_scoring
from .utils import logger, raise_error, _compute_cvmdsum
from .inspect import Inspector


def run_cross_validation(
    X: List[str],
    y: str,
    model: Union[str, PipelineCreator, BaseEstimator, List[PipelineCreator]],
    X_types: Optional[Dict] = None,
    data: Optional[pd.DataFrame] = None,
    problem_type: Optional[str] = None,
    preprocess: Union[None, str, List[str]] = None,
    return_estimator: Optional[str] = None,
    return_inspector: bool = False,
    return_train_score: bool = False,
    cv: Union[None, int] = None,
    groups: Optional[str] = None,
    scoring: Union[str, List[str], None] = None,
    pos_labels: Union[str, List[str], None] = None,
    model_params: Optional[Dict] = None,
    search_params: Optional[Dict] = None,
    seed: Optional[int] = None,
    n_jobs: Optional[int] = None,
    verbose: Optional[int] = 0,
):
    """Run cross validation and score.

    Parameters
    ----------
    X : str, list(str) or numpy.array
        The features to use.
        See :ref:`data_usage` for details.
    y : str or numpy.array
        The targets to predict.
        See :ref:`data_usage` for details.
    model : str or scikit-learn compatible model.
        If string, it will use one of the available models.
    X_types : dict[str, list of str]
        A dictionary containing keys with column type as a str and the
        columns of this column type as a list of str.
    data : pandas.DataFrame | None
        DataFrame with the data (optional).
        See :ref:`data_usage` for details.
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

    return_inspector : bool
        Whether to return the inspector object (default is False)

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
        See :ref:`data_usage` for details.
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

        See :ref:`hp_tuning` for details.
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
    inspector : Inspector | None
        The inspector object (only if ``return_inspector=True``)

    """

    # Validate parameters
    if return_estimator not in [None, "final", "cv", "all"]:
        raise_error(
            f"return_estimator must be one of None, 'final', 'cv', 'all'. "
            f"Got {return_estimator}."
        )
    if return_inspector:
        if return_estimator is None:
            logger.info("Inspector requested: setting return_estimator='all'")
            return_estimator = "all"
        if return_estimator != "all":
            raise_error(
                "return_inspector=True requires return_estimator to be `all`."
            )

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

    wrap_score = False
    if isinstance(model, (PipelineCreator, list)):
        if preprocess is not None:
            raise_error(
                "If model is a PipelineCreator (or list of), "
                "preprocess should be None"
            )
        if problem_type is not None:
            raise_error("Problem type should be set in the PipelineCreator")

        if len(model_params) > 0:
            raise_error(
                "If model is a PipelineCreator (or list of), model_params must"
                f" be None. Currently, it contains {model_params.keys()}"
            )
        if isinstance(model, list):
            if any(not isinstance(m, PipelineCreator) for m in model):
                raise_error(
                    "If model is a list, all elements must be PipelineCreator"
                )
        else:
            model = [model]

        problem_types = set([m.problem_type for m in model])
        if len(problem_types) > 1:
            raise_error(
                "If model is a list of PipelineCreator, all elements must have"
                " the same problem_type"
            )

        expanded_models = []
        for m in model:
            expanded_models.extend(m.split())

        wrap_score = expanded_models[-1]._added_target_transformer
        all_pipelines = [
            model.to_pipeline(X_types=X_types, search_params=search_params)
            for model in expanded_models
        ]

        if len(all_pipelines) > 1:
            pipeline = merge_pipelines(
                *all_pipelines, search_params=search_params
            )
        else:
            pipeline = all_pipelines[0]

        problem_type = model[0].problem_type

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
        wrap_score = pipeline_creator._added_target_transformer
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
    cv_outer = check_cv(cv, classifier=problem_type == "classification")
    logger.info(f"Using outer CV scheme {cv_outer}")

    check_consistency(y, cv, groups, problem_type)

    cv_return_estimator = return_estimator in ["cv", "all"]
    scoring = check_scoring(pipeline, scoring,
                            wrap_score=wrap_score
                            )

    cv_mdsum = _compute_cvmdsum(cv_outer)
    fit_params = {}
    if df_groups is not None:
        if isinstance(pipeline, BaseSearchCV):
            fit_params["groups"] = df_groups.values
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
        fit_params=fit_params,
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

    scores_df = pd.DataFrame(scores)
    out = scores_df
    if return_estimator in ["final", "all"]:
        pipeline.fit(df_X, y, **fit_params)
        out = scores_df, pipeline

    if return_inspector:
        inspector = Inspector(
            scores=scores_df,
            model=pipeline,
            X=df_X,
            y=y,
            groups=df_groups,
            cv=cv_outer,
        )
        out = scores_df, pipeline, inspector

    return out
