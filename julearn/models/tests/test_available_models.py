"""Provides tests for the available models registry."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
#          Shammi More <s.more@fz-juelich.de>
# License: AGPL

import pytest
from julearn.models import register_model, reset_model_register, get_model
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import warnings

# TODO: @samihamdan: Fix the ModelLike protocol to avoid all this type: ignore


def test_register_model() -> None:
    """Test the register model function."""
    register_model(
        "dt",
        classification_cls=DecisionTreeClassifier,  # type: ignore
        regression_cls=DecisionTreeRegressor,  # type: ignore
    )
    classification = get_model("dt", "classification")
    regression = get_model("dt", "regression")

    assert isinstance(classification, DecisionTreeClassifier)
    assert isinstance(regression, DecisionTreeRegressor)
    reset_model_register()

    with pytest.raises(ValueError, match="The specified model "):
        classification = get_model("dt", "classification")


def test_register_warning() -> None:
    """Test the register model function warnings."""
    with pytest.warns(RuntimeWarning, match="Model name"):
        register_model(
            "rf",
            regression_cls=RandomForestRegressor,  # type: ignore
        )
    reset_model_register()

    with pytest.raises(ValueError, match="Model name"):
        register_model(
            "rf",
            regression_cls=RandomForestRegressor,  # type: ignore
            overwrite=False,
        )
    reset_model_register()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        register_model(
            "rf",
            regression_cls=RandomForestRegressor,  # type: ignore
            overwrite=True,
        )
    reset_model_register()
