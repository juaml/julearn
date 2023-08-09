"""Implement various checks for the input of the functions."""
# Author: Federico Raimondo <f.raimondo@fz-juelich.de>
# License: BSD 3 clause

import numpy as np
import pandas as pd

from .logging import raise_error


def check_scores_df(
    *scores: pd.DataFrame, same_cv: bool = False
) -> pd.DataFrame:
    """Check the output of `run_cross_validation`.

    Parameters
    ----------
    *scores : pd.DataFrame
        DataFrames containing the scores of the models. The DataFrames must
        be the output of `run_cross_validation`
    same_cv : bool, optional
        If True, the DataFrames must have the same CV scheme, by default False

    Returns
    -------
    named_scores : list of pd.DataFrame
        The validated input DataFrames, with a `model` column added if
        missing.

    Raises
    ------
    ValueError
        If the DataFrames are not the output of `run_cross_validation` or
        if they do not have the same CV scheme and `same_cv` is True.
    """
    if any("cv_mdsum" not in x for x in scores):
        raise_error(
            "The DataFrames must be the output of `run_cross_validation`. "
            "Some of the DataFrames are missing the `cv_mdsum` column."
        )
    if any("fold" not in x for x in scores):
        raise_error(
            "The DataFrames must be the output of `run_cross_validation`. "
            "Some of the DataFrames are missing the `fold` column."
        )
    if any("repeat" not in x for x in scores):
        raise_error(
            "The DataFrames must be the output of `run_cross_validation`. "
            "Some of the DataFrames are missing the `repeat` column."
        )
    if any("n_train" not in x for x in scores):
        raise_error(
            "The DataFrames must be the output of `run_cross_validation`. "
            "Some of the DataFrames are missing the `n_train` column."
        )
    if any("n_test" not in x for x in scores):
        raise_error(
            "The DataFrames must be the output of `run_cross_validation`. "
            "Some of the DataFrames are missing the `n_test` column."
        )
    if same_cv:
        cv_mdsums = np.unique(
            np.hstack([x["cv_mdsum"].unique() for x in scores])
        )
        if cv_mdsums.size > 1:
            raise_error(
                "The CVs are not the same. Can't do a t-test on different CVs."
            )
        if cv_mdsums[0] == "non-reproducible":
            raise_error(
                "The CV is non-reproducible. Can't reproduce the CV folds."
            )

    named_scores = []
    for i, score in enumerate(scores):
        if "model" not in score:
            score["model"] = f"model_{i+1}"
        named_scores.append(score)
    return named_scores
