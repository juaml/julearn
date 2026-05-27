"""Provide tests for XGBEarlyStoppingCV."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
# License: AGPL

import pandas as pd
import pytest
from sklearn.utils.validation import _is_fitted

from julearn.models.xgb_cvearlystopping import (
    XGBClassifierCVEarlyStopping,
    XGBRegressorCVEarlyStopping,
)


def test_XGBRegressorCVEarlyStopping_grouped(df_iris) -> None:
    """Test XGBRegressorCVEarlyStopping with grouped data.

    Parameters
    ----------
    df_iris : pd.DataFrame
        The iris dataset as a DataFrame.

    """
    X = ["sepal_length", "sepal_width", "petal_width"]
    y = "petal_length"
    n_groups = 20
    bins = pd.cut(
        df_iris.index.values, labels=list(range(n_groups)), bins=n_groups
    )
    df_iris["group"] = bins.astype(int)

    model = XGBRegressorCVEarlyStopping(
        test_size=0.2, early_stopping_rounds=5, random_state=42
    )

    assert _is_fitted(model) is False
    assert not hasattr(model, "_grouped_cv")
    model.fit(df_iris[X], df_iris[y], groups=df_iris["group"])
    assert _is_fitted(model)
    assert hasattr(model, "_grouped_cv")
    assert model._grouped_cv is True
    assert model._model.get_params()["num_parallel_tree"] is None

    # Check that the model was refit with the best number of iterations
    assert model._model.get_params()["early_stopping_rounds"] is None
    assert model._model.get_params()["random_state"] == 42
    assert model._best_iteration is not None
    assert (
        model._model.get_params()["n_estimators"] == model._best_iteration + 1
    )

    y_pred = model.predict(df_iris[X])
    assert y_pred.shape == (len(df_iris),)

    score = model.score(df_iris[X], df_iris[y])
    assert isinstance(score, float)


def test_XGBRegressorCVEarlyStopping_notgrouped(df_iris) -> None:
    """Test XGBRegressorCVEarlyStopping with non-grouped data.

    Parameters
    ----------
    df_iris : pd.DataFrame
        The iris dataset as a DataFrame.

    """
    X = ["sepal_length", "sepal_width", "petal_width"]
    y = "petal_length"

    model = XGBRegressorCVEarlyStopping(
        test_size=0.2, early_stopping_rounds=5, random_state=42
    )

    assert _is_fitted(model) is False
    assert not hasattr(model, "_grouped_cv")
    model.fit(df_iris[X], df_iris[y])
    assert _is_fitted(model)
    assert hasattr(model, "_grouped_cv")
    assert model._grouped_cv is False
    assert model._model.get_params()["num_parallel_tree"] is None

    # Check that the model was refit with the best number of iterations
    assert model._model.get_params()["early_stopping_rounds"] is None
    assert model._model.get_params()["random_state"] == 42
    assert model._best_iteration is not None
    assert (
        model._model.get_params()["n_estimators"] == model._best_iteration + 1
    )


def test_XGBRegressorCVEarlyStopping_numpy(df_iris) -> None:
    """Test XGBRegressorCVEarlyStopping with numpy data.

    Parameters
    ----------
    df_iris : pd.DataFrame
        The iris dataset as a DataFrame.

    """
    X = ["sepal_length", "sepal_width", "petal_width"]
    y = "petal_length"

    model = XGBRegressorCVEarlyStopping(
        test_size=0.2,
        early_stopping_rounds=5,
        random_state=42,
        num_parallel_tree=2,
    )

    assert _is_fitted(model) is False
    assert not hasattr(model, "_grouped_cv")
    model.fit(df_iris[X].values, df_iris[y].values)
    assert _is_fitted(model)
    assert hasattr(model, "_grouped_cv")
    assert model._grouped_cv is False

    # Check that the model was refit with the best number of iterations
    assert model._model.get_params()["early_stopping_rounds"] is None
    assert model._model.get_params()["random_state"] == 42
    assert model._best_iteration is not None
    assert model._model.get_params()["num_parallel_tree"] == 2
    assert (
        model._model.get_params()["n_estimators"]
        == (model._best_iteration + 1) * 2
    )


def test_XGBClassifierCVEarlyStopping_notgrouped(df_iris) -> None:
    """Test XGBClassifierCVEarlyStopping with non-grouped data.

    Parameters
    ----------
    df_iris : pd.DataFrame
        The iris dataset as a DataFrame.

    """
    X = ["sepal_length", "sepal_width", "petal_width"]
    y = "species"

    model = XGBClassifierCVEarlyStopping(
        test_size=0.2, early_stopping_rounds=5, random_state=42
    )

    assert _is_fitted(model) is False
    assert not hasattr(model, "_grouped_cv")
    model.fit(df_iris[X], df_iris[y])
    assert _is_fitted(model)
    assert hasattr(model, "_grouped_cv")
    assert model._grouped_cv is False
    assert model._model.get_params()["num_parallel_tree"] is None

    # Check that the model was refit with the best number of iterations
    assert model._model.get_params()["early_stopping_rounds"] is None
    assert model._model.get_params()["random_state"] == 42
    assert model._best_iteration is not None

    # Three classes, so the number of trees is the best iteration times 3
    assert (
        model._model.get_params()["n_estimators"]
        == (model._best_iteration + 1) * 3
    )


def test_XGBClassifierCVEarlyStopping_grouped(df_iris) -> None:
    """Test XGBClassifierCVEarlyStopping with grouped data.

    Parameters
    ----------
    df_iris : pd.DataFrame
        The iris dataset as a DataFrame.

    """
    X = ["sepal_length", "sepal_width", "petal_width"]
    y = "species"
    n_groups = 20
    bins = pd.cut(
        df_iris.index.values, labels=list(range(n_groups)), bins=n_groups
    )
    df_iris["group"] = bins.astype(int)

    model = XGBClassifierCVEarlyStopping(
        test_size=0.2, early_stopping_rounds=5, random_state=42
    )

    assert _is_fitted(model) is False
    assert not hasattr(model, "_grouped_cv")
    model.fit(df_iris[X], df_iris[y], groups=df_iris["group"])
    assert _is_fitted(model)
    assert hasattr(model, "_grouped_cv")
    assert model._grouped_cv is True
    assert model.get_params()["test_size"] == 0.2
    assert model.get_params()["early_stopping_rounds"] == 5
    assert model.get_params()["random_state"] == 42

    # Check that the model was refit with the best number of iterations
    assert model._model.get_params()["early_stopping_rounds"] is None
    assert model._model.get_params()["random_state"] == 42
    assert model._best_iteration is not None

    # Three classes, so the number of trees is the best iteration times 3
    assert (
        model._model.get_params()["n_estimators"]
        == (model._best_iteration + 1) * 3
    )

    y_pred = model.predict(df_iris[X])
    assert y_pred.shape == (len(df_iris),)
    assert set(y_pred).issubset(set(df_iris[y]))

    y_probas = model.predict_proba(df_iris[X])
    assert y_probas.shape == (len(df_iris), 3)
    assert (y_probas >= 0).all() and (y_probas <= 1).all()

    score = model.score(df_iris[X], df_iris[y])
    assert isinstance(score, float)


def test_XGBClassifierCVEarlyStopping_binary(df_binary) -> None:
    """Test XGBClassifierCVEarlyStopping with binary classification.

    Parameters
    ----------
    df_binary : pd.DataFrame
        The binary classification dataset as a DataFrame.

    """
    X = ["sepal_length", "sepal_width", "petal_width"]
    y = "species"

    model = XGBClassifierCVEarlyStopping(
        test_size=0.2, early_stopping_rounds=5
    )

    assert _is_fitted(model) is False
    assert not hasattr(model, "_grouped_cv")
    model.fit(df_binary[X], df_binary[y])
    assert _is_fitted(model)
    assert hasattr(model, "_grouped_cv")
    assert model._grouped_cv is False
    assert model.get_params()["test_size"] == 0.2
    assert model.get_params()["early_stopping_rounds"] == 5
    # Check that the model was refit with the best number of iterations
    assert model._model.get_params()["early_stopping_rounds"] is None
    assert model._model.get_params()["random_state"] is None
    assert model._best_iteration is not None

    # Two classes, so the number of trees is the best iteration times 2
    assert (
        model._model.get_params()["n_estimators"]
        == (model._best_iteration + 1) * 2
    )
    y_pred = model.predict(df_binary[X])
    assert y_pred.shape == (len(df_binary),)
    assert set(y_pred).issubset(set(df_binary[y]))

    y_probas = model.predict_proba(df_binary[X])
    assert y_probas.shape == (len(df_binary), 2)
    assert (y_probas >= 0).all() and (y_probas <= 1).all()

    score = model.score(df_binary[X], df_binary[y])
    assert isinstance(score, float)


def test_XGBClassifierCVEarlyStopping_errors() -> None:
    """Test XGBClassifierCVEarlyStopping error handling."""
    with pytest.raises(ValueError, match="early_stopping_rounds"):
        model = XGBClassifierCVEarlyStopping(
            test_size=0.2, early_stopping_rounds=None, random_state=42
        )

    with pytest.raises(ValueError, match="not fitted"):
        model = XGBClassifierCVEarlyStopping(
            test_size=None, early_stopping_rounds=5, random_state=42
        )
        model.predict([[1, 2], [3, 4], [5, 6]])


def test_XGBClassifierCVEarlyStopping_numpy(df_iris) -> None:
    """Test XGBClassifierCVEarlyStopping with numpy data.

    Parameters
    ----------
    df_iris : pd.DataFrame
        The iris dataset as a DataFrame.

    """
    X = ["sepal_length", "sepal_width", "petal_width"]
    y = "species"

    model = XGBClassifierCVEarlyStopping(
        test_size=0.2, early_stopping_rounds=5, random_state=42
    )

    assert _is_fitted(model) is False
    assert not hasattr(model, "_grouped_cv")
    model.fit(df_iris[X].values, df_iris[y].values)
    assert _is_fitted(model)
    assert hasattr(model, "_grouped_cv")
    assert model._grouped_cv is False

    # Check that the model was refit with the best number of iterations
    assert model._model.get_params()["early_stopping_rounds"] is None
    assert model._model.get_params()["random_state"] == 42
    assert model._best_iteration is not None

    # Three classes, so the number of trees is the best iteration times 3
    assert (
        model._model.get_params()["n_estimators"]
        == (model._best_iteration + 1) * 3
    )

    model.fit(df_iris[X].values, df_iris[y].values.to_numpy())
    assert _is_fitted(model)
    assert hasattr(model, "_grouped_cv")
    assert model._grouped_cv is False

    # Check that the model was refit with the best number of iterations
    assert model._model.get_params()["early_stopping_rounds"] is None
    assert model._model.get_params()["random_state"] == 42
    assert model._best_iteration is not None

    # Three classes, so the number of trees is the best iteration times 3
    assert (
        model._model.get_params()["n_estimators"]
        == (model._best_iteration + 1) * 3
    )

    y_pred = model.predict(df_iris[X].values)
    assert y_pred.shape == (len(df_iris),)
    assert set(y_pred).issubset(set(df_iris[y]))

    y_probas = model.predict_proba(df_iris[X].values)
    assert y_probas.shape == (len(df_iris), 3)
    assert (y_probas >= 0).all() and (y_probas <= 1).all()

    score = model.score(df_iris[X].values, df_iris[y].values)
    assert isinstance(score, float)


def test_XGBClassifierCVEarlyStopping_set_params(df_iris) -> None:
    """Test XGBClassifierCVEarlyStopping with grouped data.

    Parameters
    ----------
    df_iris : pd.DataFrame
        The iris dataset as a DataFrame.

    """
    X = ["sepal_length", "sepal_width", "petal_width"]
    y = "species"
    n_groups = 20
    bins = pd.cut(
        df_iris.index.values, labels=list(range(n_groups)), bins=n_groups
    )
    df_iris["group"] = bins.astype(int)

    model = XGBClassifierCVEarlyStopping(
        test_size=0.2, early_stopping_rounds=5, random_state=42
    )

    assert _is_fitted(model) is False
    assert not hasattr(model, "_grouped_cv")
    model.fit(df_iris[X], df_iris[y], groups=df_iris["group"])
    assert _is_fitted(model)
    assert hasattr(model, "_grouped_cv")
    assert model._grouped_cv is True
    assert model.get_params()["test_size"] == 0.2
    assert model.get_params()["early_stopping_rounds"] == 5
    assert model.get_params()["random_state"] == 42

    # Check that the model was refit with the best number of iterations
    assert model._model.get_params()["early_stopping_rounds"] is None
    assert model._model.get_params()["random_state"] == 42
    assert model._best_iteration is not None

    # Three classes, so the number of trees is the best iteration times 3
    assert (
        model._model.get_params()["n_estimators"]
        == (model._best_iteration + 1) * 3
    )

    model.set_params(
        test_size=0.3,
        early_stopping_rounds=10,
        random_state=24,
        num_parallel_tree=2,
    )
    assert model.get_params()["test_size"] == 0.3
    assert model.get_params()["early_stopping_rounds"] == 10
    assert model.get_params()["random_state"] == 24
    assert model.get_params()["num_parallel_tree"] == 2
    model.fit(df_iris[X], df_iris[y], groups=df_iris["group"])
    assert _is_fitted(model)
    assert hasattr(model, "_grouped_cv")
    assert model._grouped_cv is True
    assert model.get_params()["test_size"] == 0.3
    assert model.get_params()["early_stopping_rounds"] == 10
    assert model.get_params()["random_state"] == 24
    assert model.get_params()["num_parallel_tree"] == 2
    # Check that the model was refit with the best number of iterations
    assert model._model.get_params()["early_stopping_rounds"] is None
    assert model._model.get_params()["random_state"] == 24
    assert model._best_iteration is not None
    # Three classes, so the number of trees is the best iteration times 3
    assert (
        model._model.get_params()["n_estimators"]
        == (model._best_iteration + 1) * 3 * 2
    )
