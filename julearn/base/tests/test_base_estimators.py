"""Provides tests for the base estimators."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL

from typing import List, Type

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_almost_equal
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from julearn.base import ColumnTypesLike, WrapModel
from julearn.utils.typing import ModelLike


@pytest.fixture(
    params=[
        LinearRegression,
        LogisticRegression,
        SVR,
        SVC,
        DecisionTreeRegressor,
        DecisionTreeClassifier,
    ]
)
def model(request):
    """Fixture for the models."""
    return request.param


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
@pytest.mark.parametrize(
    "apply_to,column_types,selection",
    [
        (
            None,
            ["continuous"] * 4,
            slice(0, 4),
        ),
        (
            None,
            ["continuous"] * 3 + ["cat"],
            slice(0, 3),
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
            None,
            [""] * 4,
            slice(0, 4),
        ),
        (
            ".*",
            ["continuous", "duck", "quak", "B"],
            slice(0, 4),
        ),
        (
            "*",
            ["continuous", "duck", "quak", "B"],
            slice(0, 4),
        ),
    ],
)
def test_WrapModel(
    X_iris: pd.DataFrame,  # noqa: N803
    y_iris: pd.DataFrame,
    model: Type[ModelLike],
    apply_to: ColumnTypesLike,
    column_types: List[str],
    selection: slice,
) -> None:
    """Test the WrapModel class.

    Parameters
    ----------
    X_iris : pd.DataFrame
        The iris dataset features.
    y_iris : pd.DataFrame
        The iris dataset labels.
    model : ModelLike
        The model to test.
    apply_to : ColumnTypesLike
        The column types to apply the model to.
    column_types : list of
        The column types to set in X_iris.
    selection : slice
        The columns that the apply_to selector should select.

    """
    column_types = [col or "continuous" for col in column_types]
    to_rename = {
        col: f"{col.split('__:type:__')[0]}__:type:__{ctype}"  # type:ignore
        for col, ctype in zip(X_iris.columns, column_types)
    }
    X_iris.rename(columns=to_rename, inplace=True)
    X_iris_selected = X_iris.iloc[:, selection]

    np.random.seed(42)
    lr = model()
    lr.fit(X_iris_selected, y_iris)
    pred_sk = lr.predict(X_iris_selected)

    np.random.seed(42)
    wlr = WrapModel(model(), apply_to=apply_to)
    wlr.fit(X_iris, y_iris)
    pred_ju = wlr.predict(X_iris)

    assert_almost_equal(pred_ju, pred_sk)
