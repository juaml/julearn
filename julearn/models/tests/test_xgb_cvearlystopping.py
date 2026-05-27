"""Provide tests for XGBEarlyStoppingCV."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
# License: AGPL

import pandas as pd
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
