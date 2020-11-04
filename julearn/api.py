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
        preprocess_X=['z_score'],
        preprocess_y=None,
        preprocess_confounds=['z_score'],
        return_estimator=False,
        cv=None,
        groups=None,
        scoring=None,
        pos_labels=None,
        model_selection=None,
        seed=None):
    """ Run cross validation and score.

    Parameters
    ----------
    X : np.array or string
        See https://juaml.github.io/julearn/input.html for details.
        The
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
