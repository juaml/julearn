import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_score
from .handle_inputs import (validate_data_input, read_data_input,
                            read_validate_model, read_validate_cv,
                            read_validate_user_params,
                            read_validate_preprocessing)
from .create_pipeline import create_pipeline


def run_cross_validation(
        X, y, confounds, model,
        problem_type='binary_classification', data=None,
        preprocess_X=['z_score'], preprocess_y='passthrough',
        preprocess_confounds=['z_score'], hyperparameters={},
        output_estimator=False, print_result=True,
        cv_evaluation='repeat:5_nfolds:5', cv_model_selection='same',
        seed=None):

    if seed is None:
        seed = np.random.randint(0, 50_000)
        np.random.seed(seed)
    else:
        np.random.seed(seed)

    # read_validate_data
    validate_data_input(X=X, y=y, confounds=confounds,
                        df=data)
    df_X_conf, y, confound_names = read_data_input(
        X=X, y=y, confounds=confounds, df=data)

    # read_validate preprocessing
    (preprocess_X, preprocess_y, preprocess_confounds
     ) = read_validate_preprocessing(
        preprocess_X, preprocess_y, preprocess_confounds
    )
    # read_validate model
    model_tuple = read_validate_model(
        model=model, problem_type=problem_type)

    # read_validate_cv
    cv_outer, cv_inner = read_validate_cv(cv_outer=cv_evaluation,
                                          cv_inner=cv_model_selection)
    extended_pipe = create_pipeline(preprocess_X,
                                    preprocess_y,
                                    preprocess_confounds,
                                    model_tuple, confounds,
                                    categorical_features=None)

    # read_validate_user_parameters
    hyper_params = read_validate_user_params(hyperparameters, model_tuple[0])
    # cross_validate

    grid_pipe = GridSearchCV(extended_pipe, hyper_params, cv=cv_inner)

    return cross_val_score(grid_pipe, df_X_conf, y, cv=cv_outer)


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
