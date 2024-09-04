"""Provide tests for base inspector."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL

from typing import TYPE_CHECKING

import pytest

from julearn import PipelineCreator, run_cross_validation
from julearn.inspect import Inspector


if TYPE_CHECKING:
    import pandas as pd


def test_no_cv() -> None:
    """Test inspector with no cross-validation."""
    inspector = Inspector({})  # type: ignore
    with pytest.raises(ValueError, match="No cv"):
        _ = inspector.folds


def test_no_X() -> None:
    """Test inspector with no features."""
    inspector = Inspector({}, cv=5)  # type: ignore
    with pytest.raises(ValueError, match="No X"):
        _ = inspector.folds


def test_no_y() -> None:
    """Test inspector with no targets."""
    inspector = Inspector({}, cv=5, X=[1, 2, 3])  # type: ignore
    with pytest.raises(ValueError, match="No y"):
        _ = inspector.folds


def test_no_model() -> None:
    """Test inspector with no model."""
    inspector = Inspector({})  # type: ignore
    with pytest.raises(ValueError, match="No model"):
        _ = inspector.model


def test_normal_usage(df_iris: "pd.DataFrame") -> None:
    """Test inspector.

    Parameters
    ----------
    df_iris : pd.DataFrame
        The iris dataset.

    """
    X = list(df_iris.iloc[:, :-1].columns)

    # All estimators
    out = run_cross_validation(
        X=X,
        y="species",
        data=df_iris,
        model="svm",
        return_estimator="all",
        return_inspector=True,
        problem_type="classification",
    )
    scores, pipe, inspect = out
    assert pipe == inspect.model._model  # type: ignore
    for (_, score), inspect_fold in zip(
        scores.iterrows(),  # type: ignore
        inspect.folds,  # type: ignore
    ):
        assert score["estimator"] == inspect_fold.model._model

    del pipe
    # only CV estimators
    out = run_cross_validation(
        X=X,
        y="species",
        data=df_iris,
        model="svm",
        return_estimator="cv",
        return_inspector=True,
        problem_type="classification",
    )
    scores, inspect = out
    for (_, score), inspect_fold in zip(
        scores.iterrows(),  # type: ignore
        inspect.folds,  # type: ignore
    ):
        assert score["estimator"] == inspect_fold.model._model


def test_normal_usage_with_search(df_iris: "pd.DataFrame") -> None:
    """Test inspector with search.

    Parameters
    ----------
    df_iris : pd.DataFrame
        The iris dataset.

    """
    X = list(df_iris.iloc[:, :-1].columns)

    pipe = PipelineCreator(problem_type="classification").add("svm", C=[1, 2])
    _, pipe, inspect = run_cross_validation(
        X=X,
        y="species",
        data=df_iris,
        model=pipe,
        return_estimator="all",
        return_inspector=True,
    )
    assert pipe == inspect.model._model  # type: ignore
    inspect.model.get_fitted_params()  # type: ignore
    inspect.model.get_params()  # type: ignore
