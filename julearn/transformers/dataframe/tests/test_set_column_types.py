"""Provide tests for the SetColumnTypes transformer."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL
from typing import Dict, Optional

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from julearn.transformers.dataframe import SetColumnTypes


def test_SetColumnTypes(
    X_iris: pd.DataFrame, X_types_iris: Optional[Dict]
) -> None:
    """Test SetColumnTypes.

    Parameters
    ----------
    X_iris : pd.DataFrame
        The iris dataset.
    X_types_iris : dict, optional
        The types to set in the iris dataset.
    """
    _X_types_iris = {} if X_types_iris is None else X_types_iris
    to_rename = {
        col: f"{col}__:type:__{dtype}"
        for dtype, columns in _X_types_iris.items()
        for col in columns
    }
    # Set the types
    X_iris_with_types = X_iris.rename(columns=to_rename, inplace=False)
    # Set the untyped columns to continuous
    X_iris_with_types.rename(
        columns=lambda col: (
            col if "__:type:__" in col else f"{col}__:type:__continuous"
        )
    )
    st = SetColumnTypes(X_types_iris).set_output(transform="pandas")
    Xt = st.fit_transform(X_iris)
    Xt_iris_with_types = st.fit_transform(X_iris_with_types)
    assert_frame_equal(Xt, X_iris_with_types)
    assert_frame_equal(Xt_iris_with_types, X_iris_with_types)


def test_SetColumnTypes_input_validation(X_iris: pd.DataFrame) -> None:
    """Test SetColumnTypes input validation.

    Parameters
    ----------
    X_iris : pd.DataFrame
        The iris dataset.

    """
    with pytest.raises(
        ValueError, match="Each value of X_types must be a list."
    ):
        SetColumnTypes({"confound": "chicken"}).fit(X_iris)  # type: ignore


def test_SetColumnTypes_array(
    X_iris: pd.DataFrame, X_types_iris: Optional[Dict]
) -> None:
    """Test SetColumnTypes.

    Parameters
    ----------
    X_iris : pd.DataFrame
        The iris dataset.
    X_types_iris : dict, optional
        The types to set in the iris dataset.
    """
    _X_types_iris = {} if X_types_iris is None else X_types_iris
    to_rename = {
        col: f"{icol}__:type:__{dtype}"
        for dtype, columns in _X_types_iris.items()
        for icol, col in enumerate(columns)
    }
    # Set the types
    X_iris_with_types = X_iris.rename(columns=to_rename, inplace=False)
    # Set the untyped columns to continuous
    to_rename = {
        col: f"{icol}__:type:__continuous"
        for icol, col in enumerate(X_iris.columns)
        if "__:type:__" not in col
    }
    X_iris_with_types.rename(columns=to_rename)
    st = SetColumnTypes(X_types_iris).set_output(transform="pandas")
    Xt = st.fit_transform(X_iris.values)
    Xt_iris_with_types = st.fit_transform(X_iris_with_types.values)
    assert_frame_equal(Xt, Xt_iris_with_types)
