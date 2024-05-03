"""Provides tests for the available searchers."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL

import pytest
from sklearn.model_selection import GridSearchCV

from julearn.model_selection import (
    get_searcher,
    register_searcher,
    reset_searcher_register,
)
from julearn.model_selection.available_searchers import (
    get_searcher_params_attr,
)


def test_register_searcher() -> None:
    """Test registering a searcher."""
    with pytest.raises(ValueError, match="The specified searcher "):
        get_searcher("custom_grid")
    register_searcher("custom_grid", GridSearchCV, "param_grid")
    assert get_searcher("custom_grid") == GridSearchCV

    with pytest.warns(
        RuntimeWarning, match="searcher named custom_grid already exists."
    ):
        register_searcher("custom_grid", GridSearchCV, "param_grid")

    register_searcher(
        "custom_grid", GridSearchCV, "param_grid", overwrite=True
    )
    with pytest.raises(
        ValueError, match="searcher named custom_grid already exists and "
    ):
        register_searcher(
            "custom_grid", GridSearchCV, "param_grid", overwrite=False
        )

    reset_searcher_register()


def test_reset_searcher() -> None:
    """Test resetting the searcher registry."""
    register_searcher("custom_grid", GridSearchCV, "param_grid")
    get_searcher("custom_grid")
    reset_searcher_register()
    with pytest.raises(ValueError, match="The specified searcher "):
        get_searcher("custom_grid")


@pytest.mark.parametrize(
    "searcher,expected",
    [
        ("grid", "GridSearchCV"),
        ("random", "RandomizedSearchCV"),
        ("bayes", "BayesSearchCV"),
        ("optuna", "OptunaSearchCV"),
    ],
)
def test_get_searcher(searcher: str, expected: str) -> None:
    """Test getting a searcher.

    Parameters
    ----------
    searcher : str
        The searcher name.
    expected : str
        The expected searcher class name.

    """
    out = get_searcher(searcher)
    assert out.__name__ == expected


@pytest.mark.parametrize(
    "searcher,expected",
    [
        ("grid", "param_grid"),
        ("random", "param_distributions"),
        ("bayes", "search_spaces"),
        ("optuna", "param_distributions"),
    ],
)
def test_get_searcher_params_attr(searcher: str, expected: str) -> None:
    """Test getting the params attribute of a searcher.

    Parameters
    ----------
    searcher : str
        The searcher name.
    expected : str
        The expected attribute name.

    """
    out = get_searcher_params_attr(searcher)
    assert out == expected


@pytest.mark.nodeps
def test_get_searchers_noskopt() -> None:
    """Test getting a searcher without skopt."""
    out = get_searcher("bayes")
    with pytest.raises(ImportError, match="BayesSearchCV requires"):
        out()  # type: ignore


@pytest.mark.nodeps
def test_get_searchers_nooptuna() -> None:
    """Test getting a searcher without optuna."""
    out = get_searcher("optuna")
    with pytest.raises(ImportError, match="OptunaSearchCV requires"):
        out()  # type: ignore
