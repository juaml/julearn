# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
#          Shammi More <s.more@fz-juelich.de>
# License: AGPL

import pytest
from julearn.models import register_model, reset_model_register, get_model
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import warnings

def test_register_model():
    register_model("dt",
                   classification_cls=DecisionTreeClassifier,
                   regression_cls=DecisionTreeRegressor
                   )
    classification = get_model("dt", "classification")
    regression = get_model("dt", "regression")

    assert isinstance(classification, DecisionTreeClassifier)
    assert isinstance(regression, DecisionTreeRegressor)
    reset_model_register()

    with pytest.raises(ValueError, match="The specified model "):
        classification = get_model("dt", "classification")


def test_register_warning():
    with pytest.warns(RuntimeWarning, match="Model name"):
        register_model("rf", regression_cls=RandomForestRegressor)
    reset_model_register()

    with pytest.raises(ValueError, match="Model name"):
        register_model(
            "rf", regression_cls=RandomForestRegressor, overwrite=False)
    reset_model_register()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        register_model(
            "rf", regression_cls=RandomForestRegressor, overwrite=True)
    reset_model_register()
