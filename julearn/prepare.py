# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>prepa
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL
import pandas as pd
import numpy as np

from sklearn import model_selection

from . utils import raise_error, warn, logger
from . model_selection import (RepeatedStratifiedGroupsKFold,
                               StratifiedGroupsKFold)


def _validate_input_data_df(X, y, df, groups):

    # in the dataframe
    if not isinstance(X, (str, list)):
        raise_error("X must be a string or list of strings")

    if not isinstance(y, str):
        raise_error("y must be a string")

    if not isinstance(groups, (str, type(None))):
        raise_error("groups must be a string")

    if not isinstance(df, pd.DataFrame):
        raise_error("df must be a pandas.DataFrame")

    if any(not isinstance(x, str) for x in df.columns):
        raise_error("DataFrame columns must be strings")


def _validate_input_data_df_ext(X, y, df, groups):
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


def prepare_input_data(X, y, df, pos_labels, groups):
    """Prepare the input data and variables for the pipeline

    Parameters
    ----------
    X : str, list(str)
        The features to use.
        See https://juaml.github.io/julearn/input.html for details.
    y : str
        The targets to predict.
        See https://juaml.github.io/julearn/input.html for details.
    df : pandas.DataFrame with the data.
        See https://juaml.github.io/julearn/input.html for details.
    pos_labels : str, int, float or list | None
        The labels to interpret as positive. If not None, every element from y
        will be converted to 1 if is equal or in pos_labels and to 0 if not.
    groups : str | None
        The grouping labels in case a Group CV is used.
        See https://juaml.github.io/julearn/input.         html for details.

    Returns
    -------
    df_X : pandas.DataFrame
        A dataframe with the features for each sample.
    df_y : pandas.Series
        A series with the y variable (target) for each sample.
    df_groups : pandas.Series
        A series with the grouping variable for each sample (if specified
        in the groups parameter).

    """
    logger.info("==== Input Data ====")

    # Declare them as None to avoid CI issues
    df_groups = None
    if df is None:
        raise_error("DataFrame must be provided")

    logger.info("Using dataframe as input")
    _validate_input_data_df(X, y, df, groups)
    logger.info(f"\tFeatures: {X}")
    logger.info(f"\tTarget: {y}")

    if not isinstance(X, list):
        X = [X]

    if X == [":"]:
        X_columns = [
            x for x in df.columns if x != y
        ]
        if groups is not None:
            X_columns = [x for x in X_columns if x not in groups]
    else:
        X_columns = X

    # TODO: Pick columns by regexp

    missing = [x for x in X_columns if x not in df.columns]
    if len(missing) > 0:
        raise_error(f"Missing columns in the dataframe: {missing}")

    _validate_input_data_df_ext(X_columns, y, df, groups)
    df_X = df.loc[:, X_columns].copy()
    if isinstance(df_X, pd.Series):
        df_X = df_X.to_frame()
    if y not in df.columns:
        raise_error(f"Missing target ({y}) in the dataframe")
    df_y = df.loc[:, y].copy()
    if groups is not None:
        logger.info(f"Using {groups} as groups")
        if groups not in df.columns:
            raise_error(f"Missing groups ({groups}) in the dataframe")
        df_groups = df.loc[:, groups].copy()

    if pos_labels is not None:
        if not isinstance(pos_labels, list):
            pos_labels = [pos_labels]
        logger.info(f"Setting the following as positive labels {pos_labels}")
        # TODO: Warn if pos_labels are not in df_y
        df_y = df_y.isin(pos_labels).astype(np.int64)
    logger.info("====================")
    logger.info("")
    return df_X, df_y, df_groups


def check_consistency(
    y,
    cv,
    groups,
    problem_type,
):
    """Check the consistency of the parameters/input"""

    # Check problem type and the target.
    n_classes = np.unique(y.values).shape[0]
    if problem_type == "classification":
        # If not exactly two classes:
        if n_classes != 2:
            logger.info(
                "Multi-class classification problem detected "
                f"#classes = {n_classes}."
            )
        else:
            logger.info("Binary classification problem detected.")
    else:
        # Regression
        is_numeric = np.issubdtype(y.values.dtype, np.number)
        if not is_numeric:
            warn(
                f"The kind of values in y ({y.values.dtype}) is not "
                "suitable for a regression."
            )
        else:
            n_classes = np.unique(y.values).shape[0]
            if n_classes == 2:
                warn(
                    "A regression will be performed but only 2 "
                    "distinct values are defined in y."
                )
    # Check groups and CV scheme
    if groups is not None:
        valid_instances = (
            model_selection.GroupKFold,
            model_selection.GroupShuffleSplit,
            model_selection.LeaveOneGroupOut,
            model_selection.LeavePGroupsOut,
            StratifiedGroupsKFold,
            RepeatedStratifiedGroupsKFold,
        )
        if not isinstance(cv, valid_instances):
            warn(
                "The parameter groups was specified but the CV strategy "
                "will not consider them."
            )
