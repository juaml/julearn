"""Provide tests for DynamicSelection without a compatible DESlib."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
# License: AGPL

import sys
import types

import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from julearn.models.dynamic import (
    DynamicSelection,
    _check_deslib,
    _last_incompatible_deslib,
)


def _fit_dynamic_model(
    df_iris: pd.DataFrame, ensemble_model: RandomForestClassifier
) -> None:
    """Fit a DynamicSelection model on the iris dataset.

    Parameters
    ----------
    df_iris : pd.DataFrame
        Iris dataset.
    ensemble_model : RandomForestClassifier
        The ensemble model to give to DynamicSelection.

    """
    X = ["sepal_length", "sepal_width", "petal_length"]
    y = "species"
    dynamic_model = DynamicSelection(
        ensemble=ensemble_model,  # type: ignore
        algorithm="METADES",
    )
    dynamic_model.fit(df_iris[X], df_iris[y])


def test_no_deslib(
    df_iris: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test DynamicSelection errors if DESlib is not installed.

    Parameters
    ----------
    df_iris : pd.DataFrame
        Iris dataset.
    monkeypatch : pytest.MonkeyPatch
        The monkeypatch fixture.

    """
    # A None entry in sys.modules makes the import raise ImportError
    monkeypatch.setitem(sys.modules, "deslib", None)
    with pytest.raises(ImportError, match="requires deslib"):
        _fit_dynamic_model(df_iris, RandomForestClassifier())


@pytest.mark.parametrize("version", ["0.3.5", _last_incompatible_deslib])
def test_incompatible_deslib(
    version: str, df_iris: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test DynamicSelection errors with an incompatible DESlib version.

    Parameters
    ----------
    version : str
        The DESlib version to test.
    df_iris : pd.DataFrame
        Iris dataset.
    monkeypatch : pytest.MonkeyPatch
        The monkeypatch fixture.

    """
    fake_deslib = types.ModuleType("deslib")
    fake_deslib.__version__ = version  # type: ignore
    monkeypatch.setitem(sys.modules, "deslib", fake_deslib)
    ensemble_model = RandomForestClassifier()
    with pytest.raises(RuntimeError, match="temporarily unavailable"):
        _fit_dynamic_model(df_iris, ensemble_model)
    # The error is raised before fitting anything
    assert not hasattr(ensemble_model, "estimators_")


@pytest.mark.parametrize("version", ["0.3.8", "0.4.0", "1.0"])
def test_compatible_deslib(
    version: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test the DESlib check passes with newer DESlib versions.

    Parameters
    ----------
    version : str
        The DESlib version to test.
    monkeypatch : pytest.MonkeyPatch
        The monkeypatch fixture.

    """
    fake_deslib = types.ModuleType("deslib")
    fake_deslib.__version__ = version  # type: ignore
    monkeypatch.setitem(sys.modules, "deslib", fake_deslib)
    _check_deslib()
