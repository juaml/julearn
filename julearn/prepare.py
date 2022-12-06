# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>prepa
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL
import pandas as pd
import numpy as np

from sklearn import model_selection

from . utils import raise_error, warn, logger
from . model_selection import (RepeatedStratifiedGroupsKFold,
                               StratifiedGroupsKFold)


def _validate_input_data_df(X, y, confounds, df, groups):

    # in the dataframe
    if not isinstance(X, (str, list)):
        raise_error("X must be a string or list of strings")

    if not isinstance(y, str):
        raise_error("y must be a string")

    # Confounds can be a string, list or none
    if not isinstance(confounds, (str, list, type(None))):
        raise_error(
            "If not None, confounds must be a string or list " "of strings"
        )
    if not isinstance(groups, (str, type(None))):
        raise_error("groups must be a string")

    if not isinstance(df, pd.DataFrame):
        raise_error("df must be a pandas.DataFrame")

    if any(not isinstance(x, str) for x in df.columns):
        raise_error("DataFrame columns must be strings")


def prepare_input_data(X, y, confounds, df, pos_labels, groups):
    """Prepare the input data and variables for the pipeline

    Parameters
    ----------
    X : str, list(str) or numpy.array
        The features to use.
        See https://juaml.github.io/julearn/input.html for details.
    y : str or numpy.array
        The targets to predict.
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

    """
    logger.info("==== Input Data ====")

    # Declare them as None to avoid CI issues
    df_groups = None
    if df is None:
        raise_error("TODO")

    logger.info("Using dataframe as input")
    _validate_input_data_df(X, y, confounds, df, groups)
    logger.info(f"\tFeatures: {X}")
    logger.info(f"\tTarget: {y}")

    if X == [":"]:
        X_columns = [
            x for x in df.columns if x != y
        ]
        if groups is not None:
            X_columns = [x for x in X_columns if x not in groups]
    else:
        X_columns = X

    X_conf_columns = X_columns

    df_X_conf = df.loc[:, X_conf_columns].copy()
    df_y = df.loc[:, y].copy()
    if groups is not None:
        logger.info(f"Using {groups} as groups")
        df_groups = df.loc[:, groups].copy()

    if pos_labels is not None:
        if not isinstance(pos_labels, list):
            pos_labels = [pos_labels]
        logger.info(f"Setting the following as positive labels {pos_labels}")
        # TODO: Warn if pos_labels are not in df_y
        df_y = df_y.isin(pos_labels).astype(np.int64)
    logger.info("====================")
    logger.info("")
    return df_X_conf, df_y, df_groups


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
