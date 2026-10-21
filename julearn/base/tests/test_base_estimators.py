"""Provides tests for the base estimators."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_almost_equal
from sklearn.base import clone
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from julearn.base import WrapModel
from julearn.utils.typing import ColumnTypesLike, ModelLike


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
    model: type[ModelLike],
    apply_to: ColumnTypesLike,
    column_types: list[str],
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
        for col, ctype in zip(X_iris.columns, column_types, strict=False)
    }
    X_iris.rename(columns=to_rename, inplace=True)
    X_iris_selected = X_iris.iloc[:, selection]

    np.random.seed(42)
    lr = model()
    lr.fit(X_iris_selected, y_iris)  # type: ignore
    pred_sk = lr.predict(X_iris_selected)

    np.random.seed(42)
    wlr = WrapModel(model(), apply_to=apply_to)
    wlr.fit(X_iris, y_iris)
    pred_ju = wlr.predict(X_iris)

    assert_almost_equal(pred_ju, pred_sk)


def test_WrapModel_score_sample_weight(
    X_iris: pd.DataFrame,  # noqa: N803
    y_iris: pd.DataFrame,
) -> None:
    """Test that WrapModel.score supports sample_weight.

    Parameters
    ----------
    X_iris : pd.DataFrame
        The iris dataset features.
    y_iris : pd.DataFrame
        The iris dataset labels.

    """
    weights = np.linspace(0.1, 1, len(y_iris))
    model_sk = SVR().fit(X_iris, y_iris)
    model_ju = WrapModel(SVR(), apply_to="*").fit(X_iris, y_iris)

    score_sk = model_sk.score(X_iris, y_iris)
    score_sk_weighted = model_sk.score(X_iris, y_iris, sample_weight=weights)
    assert score_sk != score_sk_weighted
    assert model_ju.score(X_iris, y_iris) == score_sk
    assert (
        model_ju.score(X_iris, y_iris, sample_weight=weights)
        == score_sk_weighted
    )

    # With metadata routing, Pipeline.score always passes sample_weight
    pipeline = Pipeline([("model", WrapModel(SVR(), apply_to="*"))])
    pipeline.fit(X_iris, y_iris)
    assert pipeline.score(X_iris, y_iris) == score_sk


def test_WrapModel_set_params() -> None:
    """Test that WrapModel.set_params follows scikit-learn."""
    model = WrapModel(SVR(), apply_to="continuous")

    # Parameters of the wrapped model and of the WrapModel
    model.set_params(C=3, apply_to="duck")
    assert model.model.C == 3
    assert model.apply_to == "duck"

    # Nested parameters of the model, as in scikit-learn
    model.set_params(model__epsilon=0.5)
    assert model.model.epsilon == 0.5

    # Replacing the model also sets the model parameters on the new model
    new_svr = SVR()
    model.set_params(C=7, model=new_svr)
    assert model.model is new_svr
    assert new_svr.C == 7

    # Nested parameters of a wrapped meta-estimator
    bagging = WrapModel(BaggingRegressor(estimator=SVR()))
    bagging.set_params(estimator__C=9)
    assert bagging.model.estimator.C == 9  # type: ignore

    # No parameters
    assert model.set_params() is model


@pytest.mark.parametrize("params", [{"svm__C": 1}, {"foo": 1}])
def test_WrapModel_set_params_invalid(params: dict) -> None:
    """Test that WrapModel.set_params raises with invalid parameters.

    Parameters
    ----------
    params : dict
        The invalid parameters to set.

    """
    model = WrapModel(SVR(), apply_to="continuous")
    with pytest.raises(ValueError, match="Invalid parameter"):
        model.set_params(**params)
    assert not hasattr(model, next(iter(params)))


def test_WrapModel_set_params_invalid_nested() -> None:
    """Test that WrapModel.set_params raises with invalid nested params."""
    model = WrapModel(SVR(), apply_to="continuous")
    with pytest.raises(ValueError, match="Invalid parameter 'foo'"):
        model.set_params(model__foo=1)


class _WrapModelExtraParam(WrapModel):
    """WrapModel with an extra parameter of its own.

    Parameters
    ----------
    model : ModelLike
        The model to wrap.
    apply_to : ColumnTypesLike, optional
        The column types to apply the model to (default None).
    needed_types : ColumnTypesLike, optional
        The column types needed by the model (default None).
    extra : int, optional
        An extra parameter of the wrapper (default 1).
    **params
        The parameters to set on the model.

    """

    def __init__(
        self,
        model: ModelLike,
        apply_to: ColumnTypesLike | None = None,
        needed_types: ColumnTypesLike | None = None,
        extra: int = 1,
        **params,
    ):
        self.extra = extra
        super().__init__(
            model, apply_to=apply_to, needed_types=needed_types, **params
        )


def test_WrapModel_set_params_own_params() -> None:
    """Test that set_params handles parameters added by a subclass."""
    model = _WrapModelExtraParam(SVR(), apply_to="continuous")
    assert "extra" in model.get_params()

    model.set_params(extra=5, C=3)
    assert model.extra == 5
    assert model.model.C == 3
    # The parameter of the wrapper is not set on the wrapped model
    assert "extra" not in model.model.get_params()

    cloned = clone(model)
    assert cloned.extra == 5
    assert cloned.model.C == 3
