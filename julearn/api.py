# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
import pandas as pd

from .prepare import (
    prepare_input_data,
    prepare_cv,
)
from .pipeline import PipelineCreator

from .utils import logger, raise_error
from .utils.typing import ModelLike


def run_cross_validation(
    X,
    y,
    model,
    X_types=None,
    data=None,
    confounds=None,
    problem_type="classification",
    preprocess=None,
    return_estimator=False,
    cv=None,
    groups=None,
    scoring=None,
    pos_labels=None,
    model_params=None,
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
    data : pandas.DataFrame | None
        DataFrame with the data (optional).
        See https://juaml.github.io/julearn/input.html for details.
    confounds : str, list(str) or numpy.array | None
        The confounds.
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

    preprocess : str, scikit-learn compatible transformers or list | None
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

    X_types = {} if X_types is None else X_types
    if seed is not None:
        # If a seed is passed, use it, otherwise do not do anything. User
        # might have set the seed outside of the library
        logger.info(f"Setting random seed to {seed}")
        np.random.seed(seed)

    # TODO: remove default CV like this
    if cv is None:
        logger.info("Using default CV")
        cv = "repeat:5_nfolds:5"

    # Interpret the input data and prepare it to be used with the library
    df_X_conf, y, df_groups, _ = prepare_input_data(
        X=X,
        y=y,
        confounds=confounds,
        df=data,
        pos_labels=pos_labels,
        groups=groups,
    )

    if model_params is None:
        model_params = {}

    search_params = {}
    if "search_params" in model_params:
        search_params = model_params.pop("search_params")

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
        pipeline = model.to_pipeline(
            X_types=X_types, search_params=search_params
        )
    elif not isinstance(model, (str, ModelLike)):
        raise_error(
            "Model has to be a PipelineCreator, a string or a "
            "scikit-learn compatible model."
        )
    else:
        if isinstance(preprocess, str):
            preprocess = [preprocess]
        if isinstance(preprocess, list):
            pipeline_creator = PipelineCreator.from_list(
                preprocess, model_params
            )
        elif preprocess is None:
            pipeline_creator = PipelineCreator()
        elif not isinstance(preprocess, PipelineCreator):
            raise_error(
                "preprocess has to be a PipelineCreator, a string or a "
                "list of strings."
            )
        else:
            pipeline_creator = preprocess

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
        pipeline_creator.add(model, problem_type=problem_type, **t_params)

        pipeline = pipeline_creator.to_pipeline(
            X_types=X_types, search_params=search_params
        )

    # Prepare cross validation
    cv_outer = prepare_cv(cv)

    # check_consistency(pipeline, preprocess_X, preprocess_y,
    #                   preprocess_confounds, df_X_conf, y, cv, groups,
    #                   problem_type)

    cv_return_estimator = return_estimator in ["cv", "all"]

    scores = cross_validate(
        pipeline,
        df_X_conf,
        y,
        cv=cv_outer,
        scoring=scoring,
        groups=df_groups,
        return_estimator=cv_return_estimator,
        n_jobs=n_jobs,
    )

    n_repeats = getattr(cv_outer, "n_repeats", 1)
    n_folds = len(scores["fit_time"]) // n_repeats

    repeats = np.repeat(np.arange(n_repeats), n_folds)
    folds = np.tile(np.arange(n_folds), n_repeats)

    scores["repeat"] = repeats
    scores["fold"] = folds

    out = pd.DataFrame(scores)
    if return_estimator in ["final", "all"]:
        pipeline.fit(df_X_conf, y)
        out = out, pipeline

    return out
