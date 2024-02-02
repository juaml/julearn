"""Provide functions to inspect the preprocessing steps of pipeline."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL

from typing import List, Optional

import pandas as pd
from sklearn.pipeline import Pipeline

from ..utils import raise_error


def preprocess(
    pipeline: Pipeline,
    X: List[str],  # noqa: N803
    data: pd.DataFrame,
    until: Optional[str] = None,
    with_column_types: bool = False,
) -> pd.DataFrame:
    """Preprocess data with a pipeline until a certain step (inclusive).

    Parameters
    ----------
    pipeline : Pipeline
        The pipeline to use.
    X : list of str
        The features to use.
    data : pd.DataFrame
        The data to preprocess.
    until : str, optional
        The name of the step to preprocess until (inclusive). If None, will
        preprocess all steps (default is None).
    with_column_types : bool, optional
        Whether to include the column types in the output (default is False).

    Returns
    -------
    pd.DataFrame
        The preprocessed data.

    """
    _X = data[X]
    if until is None:
        i = -1
    else:
        i = 1
        for name, _ in pipeline.steps[:-1]:
            if name == until:
                break
            i += 1
        else:
            raise_error(f"No step named {until} found.")
    df_out = pipeline[:i].transform(_X)

    if not isinstance(df_out, pd.DataFrame) and with_column_types is False:
        raise_error(
            "The output of the pipeline is not a DataFrame. Cannot remove "
            "column types."
        )
    if not with_column_types:
        rename_dict = {
            col: col.split("__:type:__")[0]  # type: ignore
            for col in df_out.columns
        }
        df_out.rename(columns=rename_dict, inplace=True)
    return df_out
