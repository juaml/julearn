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


def test_register_searcher() -> None:
    """Test registering a searcher."""
    with pytest.raises(ValueError, match="The specified searcher "):
        get_searcher("custom_grid")
    register_searcher("custom_grid", GridSearchCV)
    assert get_searcher("custom_grid") == GridSearchCV

    with pytest.warns(
        RuntimeWarning, match="searcher named custom_grid already exists."
    ):
        register_searcher("custom_grid", GridSearchCV)

    register_searcher("custom_grid", GridSearchCV, overwrite=True)
    with pytest.raises(
        ValueError, match="searcher named custom_grid already exists and "
    ):
        register_searcher("custom_grid", GridSearchCV, overwrite=False)

    reset_searcher_register()


def test_reset_searcher() -> None:
    """Test resetting the searcher registry."""
    register_searcher("custom_grid", GridSearchCV)
    get_searcher("custom_grid")
    reset_searcher_register()
    with pytest.raises(ValueError, match="The specified searcher "):
        get_searcher("custom_grid")
