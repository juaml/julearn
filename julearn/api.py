import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_score

from . prepare import (prepare_input_data,
                       prepare_model,
                       prepare_cv,
                       prepare_hyperparams,
                       prepare_preprocessing,
                       prepare_scoring)
from . pipeline import create_pipeline
from . model_selection import wrap_search


def run_cross_validation(
        X, y, model,
        data=None,
        confounds=None,
        problem_type='binary_classification',
        preprocess_X=['z_score'],
        preprocess_y=None,
        preprocess_confounds=['z_score'],
        hyperparameters=None,
        output_estimator=False, print_result=True,
        cv_evaluation='repeat:5_nfolds:5',
        cv_model_selection='same',
        scoring=None,
        pos_labels=None,
        seed=None):
    """ Run cross validation and score
    """

    if seed is not None:
        # If a seed is passed, use it, otherwise do not do anything. User
        # might have set the seed outside of the library
        np.random.seed(seed)

    # Interpret the input data and prepare it to be used with the library
    df_X_conf, y, confound_names = prepare_input_data(
        X=X, y=y, confounds=confounds, df=data, pos_labels=pos_labels)

    # Interpret preprocessing parameters
    preprocess_vars = prepare_preprocessing(
        preprocess_X, preprocess_y, preprocess_confounds
    )
    preprocess_X, preprocess_y, preprocess_confounds = preprocess_vars

    # Prepare the model
    model_tuple = prepare_model(model=model, problem_type=problem_type)

    # Prepare cross validation
    cv_outer, cv_inner = prepare_cv(cv_outer=cv_evaluation,
                                    cv_inner=cv_model_selection)

    pipeline = create_pipeline(preprocess_X,
                               preprocess_y,
                               preprocess_confounds,
                               model_tuple, confounds,
                               categorical_features=None)

    if hyperparameters is not None:
        # read_validate_user_parameters
        hyper_params = prepare_hyperparams(
            hyperparameters, pipeline, model_tuple[0])

        # cross_validate
        if len(hyper_params) > 0:
            pipeline = wrap_search(
                GridSearchCV, pipeline, hyper_params, cv=cv_inner)

    scorer = prepare_scoring(pipeline, scoring)
    scores = cross_val_score(pipeline, df_X_conf, y, cv=cv_outer,
                             scoring=scorer)
    return scores


def pprint_output(out):
    replication_dict = out["replication_dict"]

    replication_dict["is_classification"] = (
        "Classification"
        if replication_dict["is_classification"]
        else "Regression"
    )

    replication_dict = out["replication_dict"]

    replication_dict["is_classification"] = (
        "Classification"
        if replication_dict["is_classification"]
        else "Regression"
    )
    print(
        """
    ### Settings ###
    This is a {is_classification} Problem\n

    Using Features: {X}
    Predicting Target: {y}
    With Confounds : {confounds}

    Cross-validation settings : repeats={n_repeats} & folds={n_folds}
    Using Random Seed: {seed}

    """.format(
            **replication_dict
        )
    )

    out["fold_wise_scores"] = [round(i, 2) for i in out["fold_wise_scores"]]

    print(
        """
    ### Results ###
    Over all Folds Score: Mean={mean_scores:.2f}, SD={std_scores:.2f}

    Score per Fold:


    {fold_wise_scores}
    """.format(
            **out
        )
    )
