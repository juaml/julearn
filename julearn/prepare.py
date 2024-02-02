"""Prepare and validate the input data and parameters for the pipeline."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>prepa
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL

import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    BaseCrossValidator,
    BaseShuffleSplit,
    GroupKFold,
    GroupShuffleSplit,
    LeaveOneGroupOut,
    LeavePGroupsOut,
    StratifiedGroupKFold,
)
from sklearn.model_selection._split import _RepeatedSplits

from .config import get_config
from .model_selection import (
    ContinuousStratifiedGroupKFold,
    RepeatedContinuousStratifiedGroupKFold,
)
from .utils import logger, raise_error, warn_with_log


def _validate_input_data_df(
    X: Union[str, List[str]],  # noqa: N803
    y: str,
    df: pd.DataFrame,
    groups: Optional[str],
) -> None:
    """Validate the input data types for the pipeline.

    Parameters
    ----------
    X : str, list(str)
        The features to use.
        See :ref:`data_usage` for details.
    y : str
        The targets to predict.
        See :ref:`data_usage` for details.
    df : pandas.DataFrame with the data.
        See :ref:`data_usage` for details.
    groups : str | None
        The grouping labels in case a Group CV is used.
        See :ref:`data_usage` for details.

    Raises
    ------
    ValueError
        If any of the input parameters is not of a valid type.

    """
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


def _validate_input_data_df_ext(
    X: Union[str, List[str]],  # noqa: N803
    y: str,
    df: pd.DataFrame,
    groups: Optional[str],
) -> None:
    """Validate the input dataframe for the pipeline.

    Parameters
    ----------
    X : str, list(str)
        The features to use.
        See :ref:`data_usage` for details.
    y : str
        The targets to predict.
        See :ref:`data_usage` for details.
    df : pandas.DataFrame with the data.
        See :ref:`data_usage` for details.
    groups : str | None
        The grouping labels in case a Group CV is used.
        See :ref:`data_usage` for details.

    Raises
    ------
    ValueError
        If the columns specified in X, y or groups are not in the dataframe or
        they are not valid.

    """
    if not get_config("disable_x_check"):
        missing_columns = [t_x for t_x in X if t_x not in df.columns]
        # In reality, this is not needed as the regexp match will fail
        # Leaving it as additional check in case the regexp match changes
        if len(missing_columns) > 0:  # pragma: no cover
            raise_error(  # pragma: no cover
                "All elements of X must be in the dataframe. "
                f"The following are missing: {missing_columns}"
            )

    if y not in df.columns:
        raise_error(f"Target '{y}' (y) is not a valid column in the dataframe")

    if groups is not None:
        if groups not in df.columns:
            raise_error(
                f"Groups '{groups}' is not a valid column in the dataframe"
            )
        if groups == y:
            warn_with_log("y and groups are the same column")
        if groups in X:
            warn_with_log("groups is part of X")

    if y in X:
        warn_with_log(f"List of features (X) contains the target {y}")


def _is_regex(string: str) -> bool:
    """Check if a string is a regular expression.

    Parameters
    ----------
    string : str
        The string to check.

    Returns
    -------
    bool
        True if the string should be interpreted as a regular expression,
        False otherwise.

    """
    _regex_chars = ["*", "+", "?", ".", "|"]
    if any(char in string for char in _regex_chars):
        return True
    return False


def _pick_columns(
    regexes: Union[str, List[str]], columns: Union[List[str], pd.Index]
) -> List[str]:
    """Pick elements from a list based on matches to a list of regexes.

    Parameters
    ----------
    regexes : str or list(str)
        List of regular expressions to match
    columns : list(str)
        Elements to pick

    Returns
    -------
    picks : list(str)
        A list will all the elements from columns that match at least one
        regexp in regexes

    Raises
    ------
    ValueError
        If one or more regexes do not match any element in columns

    """
    if not isinstance(regexes, list):
        regexes = [regexes]

    picks = []
    for exp in regexes:
        if not _is_regex(exp):
            if exp in columns:
                picks.append(exp)
        else:
            cols = [
                col
                for col in columns
                if any([re.fullmatch(exp, col)]) and col not in picks
            ]
            if len(cols) > 0:
                picks.extend(cols)

    skip_this_check = get_config("disable_x_check")
    if not skip_this_check:
        if len(columns) > get_config("MAX_X_WARNS"):
            warn_with_log(
                f"The dataframe has {len(columns)} columns. Checking X for "
                "consistency might take a while. To skip this checks, set the "
                "config flag `disable_x_check` to `True`."
            )
        unmatched = []
        for exp in regexes:
            if not any(re.fullmatch(exp, col) for col in columns):
                unmatched.append(exp)
        if len(unmatched) > 0:
            raise ValueError(
                "All elements must be matched. "
                f"The following are missing: {unmatched}"
            )
    return picks


def prepare_input_data(
    X: Union[str, List[str]],  # noqa: N803
    y: str,
    df: pd.DataFrame,
    pos_labels: Union[str, int, float, List, None],
    groups: Optional[str],
    X_types: Optional[Dict],  # noqa: N803
) -> Tuple[pd.DataFrame, pd.Series, Union[pd.Series, None], Dict]:
    """Prepare the input data and variables for the pipeline.

    Parameters
    ----------
    X : str, list(str)
        The features to use.
        See :ref:`data_usage` for details.
    y : str
        The targets to predict.
        See :ref:`data_usage` for details.
    df : pandas.DataFrame with the data.
        See :ref:`data_usage` for details.
    pos_labels : str, int, float or list | None
        The labels to interpret as positive. If not None, every element from y
        will be converted to 1 if is equal or in pos_labels and to 0 if not.
    groups : str | None
        The grouping labels in case a Group CV is used.
        See :ref:`data_usage` for details.
    X_types : dict | None
        A dictionary containing keys with column type as a str and the
        columns of this column type as a list of str.

    Returns
    -------
    df_X : pandas.DataFrame
        A dataframe with the features for each sample.
    df_y : pandas.Series
        A series with the y variable (target) for each sample.
    df_groups : pandas.Series
        A series with the grouping variable for each sample (if specified
        in the groups parameter).

    Raises
    ------
    ValueError
        If there is any error on the input data and parameters validation.

    Warns
    -----
    RuntimeWarning
        If the input data and parameters might have inconsistencies.

    """
    logger.info("==== Input Data ====")

    # Declare them as None to avoid CI issues
    df_groups = None

    logger.info("Using dataframe as input")

    # Validate the variables
    _validate_input_data_df(X, y, df, groups)
    logger.info(f"\tFeatures: {X}")
    logger.info(f"\tTarget: {y}")

    if not isinstance(X, list):
        X = [X]

    if X == [":"]:
        X_columns = [x for x in df.columns if x != y]
        if groups is not None:
            X_columns = [x for x in X_columns if x not in groups]
    else:
        X_columns = _pick_columns(X, df.columns)

    if not get_config("disable_x_verbose"):
        logger.info(f"\tExpanded features: {X_columns}")

    _validate_input_data_df_ext(X_columns, y, df, groups)

    X_types = _check_x_types(X_types, X_columns)

    # Get X
    df_X = df.loc[:, X_columns].copy()
    if isinstance(df_X, pd.Series):
        df_X = df_X.to_frame()

    # Get y
    df_y = df.loc[:, y].copy()

    # Get groups
    if groups is not None:
        logger.info(f"Using {groups} as groups")
        df_groups = df.loc[:, groups].copy()

    # Convert the target to binary if pos_labels is not None
    if pos_labels is not None:
        if not isinstance(pos_labels, list):
            pos_labels = [pos_labels]
        logger.info(f"Setting the following as positive labels {pos_labels}")

        not_in_labels = [x for x in pos_labels if x not in df_y.unique()]
        if len(not_in_labels) > 0:
            warn_with_log(
                f"The following labels are not in the target: {not_in_labels}"
            )

        df_y = df_y.isin(pos_labels).astype(int)
        if 0 not in df_y.unique():
            warn_with_log(
                "All targets have been set to 1. Check the pos_labels "
                "argument or your data."
            )
        if 1 not in df_y.unique():
            warn_with_log(
                "All targets have been set to 0. Check the pos_labels "
                "argument or your data."
            )

    logger.info("====================")
    logger.info("")
    return df_X, df_y, df_groups, X_types


def check_consistency(
    y: pd.Series,
    cv: Union[int, BaseCrossValidator, BaseShuffleSplit, _RepeatedSplits],
    groups: Optional[pd.Series],
    problem_type: str,
) -> None:
    """Check the consistency of the parameters/input.

    Parameters
    ----------
    y : pandas.Series
        The target variable.
    cv : int or cross validator object
        The cross-validator to use.
    groups : pandas.Series | None
        The grouping variable.
    problem_type : str
        The problem type. Can be "classification" or "regression".

    Raises
    ------
    ValueError
        If there is any inconsistency between the parameters and the data.

    Warns
    -----
    RuntimeWarning
        If there might be an inconsistency between the parameters and the data
        but the pipeline can still run.

    """

    # Check problem type and the target.
    n_classes = np.unique(y.values).shape[0]  # type: ignore
    if problem_type == "classification":
        # If not exactly two classes:
        if n_classes == 1:
            raise_error(
                "There is only one class in y. Check the target variable.",
                ValueError,
            )
        if n_classes != 2:
            logger.info(
                "Multi-class classification problem detected "
                f"#classes = {n_classes}."
            )
            if n_classes > (y.shape[0] / 2):
                warn_with_log(
                    f"The number of classes ({n_classes}) is larger than the "
                    "number of samples divided by 2. This might be the cause "
                    "of a wrong problem type. Are you sure you want to do a "
                    "multi-class classification?"
                )
        else:
            logger.info("Binary classification problem detected.")
    else:
        # Regression
        is_numeric = np.issubdtype(y.values.dtype, np.number)  # type: ignore
        if not is_numeric:
            warn_with_log(
                f"The kind of values in y ({y.values.dtype}) is not "
                "suitable for a regression."
            )
        else:
            n_classes = np.unique(y.values).shape[0]  # type: ignore
            if n_classes == 2:
                warn_with_log(
                    "A regression will be performed but only 2 "
                    "distinct values are defined in y."
                )
    # Check groups and CV scheme
    if groups is not None:
        valid_instances = (
            GroupKFold,
            GroupShuffleSplit,
            LeaveOneGroupOut,
            LeavePGroupsOut,
            StratifiedGroupKFold,
            ContinuousStratifiedGroupKFold,
            RepeatedContinuousStratifiedGroupKFold,
        )
        if not isinstance(cv, valid_instances):
            warn_with_log(
                "The parameter groups was specified but the CV strategy "
                "will not consider them."
            )


def _check_x_types(
    X_types: Optional[Dict], X: List[str]  # noqa: N803
) -> Dict[str, List]:
    """Check validity of X_types with respect to X.

    Parameters
    ----------
    X_types : dict, optional
        A dictionary with the types of the features. If None, an empty
        dictionary is returned.
    X : list
        A list with the names of the features.

    Returns
    -------
    X_types : dict[str, list]
        A dictionary with the types of the features, ready to use

    Raises
    ------
    ValueError
        If there are columns in X_types that are not defined in X.

    Warns
    -----
    RuntimeWarning
        If there are columns in X that are not defined in X_types or if X_types
        is None.

    """
    if X_types is None:
        warn_with_log(
            "X_types is None. No type checking will be performed and all "
            "features will be treated as continuous."
        )
        return {}
    X_types = {
        k: [v] if not isinstance(v, list) else v for k, v in X_types.items()
    }
    if not get_config("disable_xtypes_verbose"):
        logger.info(f"\tX_types:{X_types}")

    skip_this_check = get_config("disable_xtypes_check")
    if not skip_this_check:
        if len(X) > get_config("MAX_X_WARNS"):
            warn_with_log(
                f"X has {len(X)} columns. Checking X_types for consistency "
                "might take a while. To skip this checks, set the config flag "
                "`disable_xtypes_check` to `True`."
            )
        missing_columns = []
        defined_columns = []
        for _, columns in X_types.items():
            t_columns = [
                col
                for col in X
                if any(re.fullmatch(exp, col) for exp in columns)
            ]
            t_missing = [
                exp
                for exp in columns
                if not any(re.fullmatch(exp, col) for col in X)
            ]
            defined_columns.extend(t_columns)
            missing_columns.extend(t_missing)
        duplicated_columns = [
            k for k, v in Counter(defined_columns).items() if v > 1
        ]
        if len(duplicated_columns) > 0:
            raise_error(
                "The following columns are defined more than once in X_types: "
                f"{duplicated_columns}",
                ValueError,
            )
        undefined_columns = [x for x in X if x not in defined_columns]

        if len(missing_columns) > 0:
            raise_error(
                "The following columns are defined in X_types but not in X: "
                f"{missing_columns}",
                ValueError,
            )
        if len(undefined_columns) > 0:
            warn_with_log(
                f"The following columns are not defined in X_types: "
                f"{undefined_columns}. They will be treated as continuous."
            )
    return X_types


def prepare_search_params(
    search_params: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Prepare the parameters for the search.

    Parameters
    ----------
    search_params : dict, optional
        The parameters for the search.

    Returns
    -------
    search_params : dict
        The parameters for the search, ready to use.

    """

    if search_params is None:
        search_params = {}
    else:
        search_params = search_params.copy()

    if "cv" not in search_params:
        search_params["cv"] = None
    if "kind" not in search_params:
        search_params["kind"] = "grid"

    return search_params
