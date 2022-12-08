"""Provide tests for validating column names in X_types."""

from julearn.prepare import check_x_types
import pytest


def test_check_x_types_valid():
    """Test check_columns if all X_types are valid."""
    X = ['a', 'b', 'c', 'd']
    X_types = {'type1': ['a', 'b'], 'type2': ['c', 'd']}
    check_x_types(X_types, X)


def test_check_x_types_duplicate():
    """Test check_columns if there are duplicate column names in X_types."""
    X = ['a', 'b', 'c', 'd']
    X_types = {'type1': ['a', 'b', 'c'], 'type2': ['c', 'd']}
    with pytest.raises(
        ValueError, match='One column defined multiple times in X_types'
    ):
        check_x_types(X_types, X)


def test_check_x_types_invalid():
    """Test if there is a name in X_types that does not exist in X."""
    X = ['a', 'b', 'c', 'd']
    X_types = {'type1': ['a', 'b', 'e'], 'type2': ['c', 'd']}
    with pytest.raises(
        ValueError, match='e of type type1 not a valid column name in X.'
    ):
        check_x_types(X_types, X)
