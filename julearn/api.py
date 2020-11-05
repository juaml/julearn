import numpy as np
from sklearn.model_selection import cross_val_score

from . prepare import (prepare_input_data,
                       prepare_model,
                       prepare_cv,
                       prepare_model_selection,
                       prepare_preprocessing,
                       prepare_scoring)
from . pipeline import create_pipeline

from . utils import logger


def run_cross_validation(
        X, y, model,
        data=None,
        confounds=None,
        problem_type='binary_classification',
        preprocess_X='zscore',
        preprocess_y=None,
        preprocess_confounds='zscore',
        return_estimator=False,
        cv=None,
        groups=None,
        scoring=None,
        pos_labels=None,
        model_selection=None,
        seed=None):
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
          in which the target (y) has only two posible classes (default).
          The parameter pos_labels can be used to convert a target with
          multiple_classes into binary.
        * "multiclass_classification": Performs a multiclass classification
          in which the target (y) has more than two possible values.
        * "regression". Perform a regression. The target (y) has to be
          ordinal at least.

    preprocess_X : str, scikit-learn compatible transformers or list | None
        Transformer to apply to the features (X). If string, use one of the
        available transformers. If list, each element can be a string or
        scikit-learn compatible transformer. If None, no transformation
        is applied.

        Defaults to zscore (StandardScaler).

        See https://juaml.github.io/julearn/pipeline.html for details.
    preprocess_y : str or scikit-learn transformer | None
        Transformer to apply to the target (y). If None, no transformation
        is applied.

        See https://juaml.github.io/julearn/pipeline.html for details.
    preprocess_confounds : str, scikit-learn transformers or list | None
        Transformer to apply to the features (X). If string, use one of the
        available transformers. If list, each element can be a string or
        scikit-learn compatible transformer. If None, no transformation
        is applied.

        Defaults to zscore (StandardScaler).

        See https://juaml.github.io/julearn/pipeline.html for details.
    return_estimator : bool
        If True, it returns the estimator (pipeline) fitted on all the data.
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
    scoring : str | None
        The scoring metric to use.
        See https://scikit-learn.org/stable/modules/model_evaluation.html for
        a comprehensive list of options. If None, use 'accuracy'.
    pos_labels : str, int, float or list | None
        The labels to interpret as positive. If not None, every element from y
        will be converted to 1 if is equal or in pos_labels and to 0 if not.
    model_selection : dict | None
        If not None, this dictionary specifies the parameters to use for model
        hyperparameters and selection of hyperparameters.

        The dictionary can define the following keys:

        * 'hyperparameters': A dictionary setting hyperparameters for each
          step of the pipeline. If more than option is provided for at least
          one hyperparameter, a GridSearch will be performed.
        * 'cv': If GridSearch is going to be used, the cross-validation
          splitting stategy to use. Defaults to same CV as for the model
          evaluation.
        * 'gs_scoring': If GridSearch is going to be used, the scoring metric
          to evaluate the performance.

        See https://juaml.github.io/julearn/hyperparameters.html for details.
    seed : int | None
        If not None, set the random seed before any operation. Usefull for
        reproducibility.

    Returns
    -------
    scores : np.ndarray
        The fold-wise score.
    estimator : object
        The estimator, fitted on all the data (only if
        ``return_estimator=True``)

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

    # Interpret preprocessing parameters
    preprocess_vars = prepare_preprocessing(
        preprocess_X, preprocess_y, preprocess_confounds
    )
    preprocess_X, preprocess_y, preprocess_confounds = preprocess_vars

    # Prepare the model
    model_tuple = prepare_model(model=model, problem_type=problem_type)

    # Prepare cross validation
    cv_outer = prepare_cv(cv)

    pipeline = create_pipeline(preprocess_X,
                               preprocess_y,
                               preprocess_confounds,
                               model_tuple, confounds,
                               categorical_features=None)

    if model_selection is not None:
        pipeline = prepare_model_selection(
            model_selection, pipeline, model_tuple[0], cv_outer)

    scorer = prepare_scoring(pipeline, scoring)
    scores = cross_val_score(pipeline, df_X_conf, y, cv=cv_outer,
                             scoring=scorer, groups=df_groups)

    out = scores
    if return_estimator is True:
        pipeline.fit(df_X_conf, y)
        out = out, pipeline

    return out
