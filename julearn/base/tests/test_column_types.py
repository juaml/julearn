"""Provides tests for ColumnTypes."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL

from typing import List, Set

import pandas as pd
import pytest

from julearn.base import ColumnTypes, ColumnTypesLike, make_type_selector


@pytest.mark.parametrize(
    "pattern,column_types,selection",
    [
        (
            "(?:__:type:__continuous)",
            ["continuous"] * 4,
            slice(0, 4),
        ),
        (
            "(?:__:type:__continuous)",
            ["continuous"] * 3 + ["cat"],
            slice(0, 3),
        ),
        (
            "(?:__:type:__cont|__:type:__cat)",
            ["cont"] * 3 + ["cat"],
            slice(0, 4),
        ),
        (
            "(?:__:type:__continuous)",
            [""] * 4,
            slice(0, 4),
        ),
        (
            ".*",
            ["continuous", "duck", "quak", "B"],
            slice(0, 4),
        ),
    ],
)
def test_make_column_selector(
    X_iris: pd.DataFrame,  # noqa: N803
    pattern: str,
    column_types: List[str],
    selection: slice,
) -> None:
    """Test the make_column_selector function.

    Parameters
    ----------
    X_iris : pd.DataFrame
        The iris dataset features.
    pattern : str
        The pattern to test.
    column_types : list of
        The column types to set in X_iris.
    selection : slice
        The columns that the selector should select.

    """
    column_types = [col or "continuous" for col in column_types]
    to_rename = {
        col: f"{col.split('__:type:__')[0]}__:type:__{ctype}"  # type:ignore
        for col, ctype in zip(X_iris.columns, column_types)
    }
    X_iris.rename(columns=to_rename, inplace=True)
    col_true_selected = X_iris.iloc[:, selection].columns.tolist()
    col_selected = make_type_selector(pattern)(X_iris)
    assert col_selected == col_true_selected


@pytest.mark.parametrize(
    "column_types,pattern,resulting_column_types",
    [
        (["continuous"], "(?:__:type:__continuous)", {"continuous"}),
        ("continuous", "(?:__:type:__continuous)", {"continuous"}),
        (
            ColumnTypes("continuous"),
            "(?:__:type:__continuous)",
            {"continuous"},
        ),
        (
            ["continuous", "categorical"],
            [
                "(?:__:type:__continuous|__:type:__categorical)",
                "(?:__:type:__categorical|__:type:__continuous)",
            ],
            {"continuous", "categorical"},
        ),
        (
            ColumnTypes(["continuous", "categorical"]),
            [
                "(?:__:type:__continuous|__:type:__categorical)",
                "(?:__:type:__categorical|__:type:__continuous)",
            ],
            {"continuous", "categorical"},
        ),
        ("*", ".*", {"*"}),
        (["*"], ".*", {"*"}),
        (".*", ".*", {".*"}),
        ([".*"], ".*", {".*"}),
    ],
)
def test_ColumnTypes_patterns(
    column_types: ColumnTypesLike,
    pattern: List[str],
    resulting_column_types: Set[str],
) -> None:
    """Test the ColumnTypes patterns.

    Parameters
    ----------
    column_types : list of str
        The column types to test.
    pattern : list of str
        The patterns that should match the column types.
    resulting_column_types : set of str
        The resulting column types.

    """
    ct = ColumnTypes(column_types)
    if not isinstance(pattern, list):
        pattern = [pattern]
    assert any(ct.pattern == x for x in pattern)
    assert ct._column_types == resulting_column_types


@pytest.mark.parametrize(
    "selected_column_types,data_column_types,selection",
    [
        (
            ["continuous"],
            ["continuous"] * 4,
            slice(0, 4),
        ),
        (
            ["continuous"],
            ["continuous"] * 3 + ["cat"],
            slice(0, 3),
        ),
        (
            ["cont", "cat"],
            ["cont"] * 3 + ["cat"],
            slice(0, 4),
        ),
        (
            ["continuous"],
            [""] * 4,
            slice(0, 4),
        ),
        (
            ".*",
            ["continuous", "duck", "quak", "B"],
            slice(0, 4),
        ),
    ],
)
def test_ColumnTypes_to_column_selector(
    X_iris: pd.DataFrame,  # noqa: N803
    selected_column_types: ColumnTypesLike,
    data_column_types: List[str],
    selection: slice,
) -> None:
    """Test the ColumnTyes.to_column_selector method.

    Parameters
    ----------
    X_iris : pd.DataFrame
        The iris dataset features.
    pattern : str
        The pattern to test.
    selected_column_types : list of
        The column types to set in X_iris.
    data_column_types : set of str
        The resulting column types.
    selection : slice
        The columns that the selector should select.

    """
    _column_types = [col or "continuous" for col in data_column_types]
    to_rename = {
        col: f"{col.split('__:type:__')[0]}__:type:__{ctype}"  # type:ignore
        for col, ctype in zip(X_iris.columns, _column_types)
    }
    X_iris.rename(columns=to_rename, inplace=True)
    col_true_selected = X_iris.iloc[:, selection].columns.tolist()
    col_selected = ColumnTypes(selected_column_types).to_type_selector()(
        X_iris
    )
    assert col_selected == col_true_selected


@pytest.mark.parametrize(
    "left,right,equal",
    [
        (ColumnTypes(["continuous"]), ["continuous"], True),
        (ColumnTypes(["continuous"]), "continuous", True),
        (ColumnTypes(["continuous"]), ColumnTypes("continuous"), True),
        (ColumnTypes(["continuous", "cat"]), ["continuous", "cat"], True),
        (ColumnTypes(["continuous", "cat"]), "continuous", False),
        (ColumnTypes(["cont", "cat"]), ColumnTypes("continuous"), False),
    ],
)
def test_ColumnTypes_equivalence(
    left: ColumnTypesLike, right: ColumnTypesLike, equal: bool
) -> None:
    """Test the ColumnTypes equivalence.

    Parameters
    ----------
    left : ColumnTypesLike
        The left hand side of the comparison.
    right : ColumnTypesLike
        The right hand side of the comparison.
    equal : bool
        Whether the comparison should be equal.

    """
    assert (left == right) == equal


@pytest.mark.parametrize(
    "left,right,result",
    [
        (
            ["continuous"],
            ["continuous"],
            ["continuous"],
        ),
        (
            ["cont"],
            "cat",
            ["cont", "cat"],
        ),
    ],
)
def test_ColumnTypes_add(
    left: ColumnTypesLike, right: ColumnTypesLike, result: ColumnTypesLike
) -> None:
    """Test the ColumnTypes addition.

    Parameters
    ----------
    left : ColumnTypesLike
        The left hand side of the addition.
    right : ColumnTypesLike
        The right hand side of the addition.
    result : ColumnTypes
        The expected result.

    """
    summed = ColumnTypes(left).add(right)
    assert summed == ColumnTypes(result)
