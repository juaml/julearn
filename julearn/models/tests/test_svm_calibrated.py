"""Provide tests for the calibrated Support Vector Classifier."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
# License: AGPL

import warnings

import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC

from julearn import run_cross_validation
from julearn.models import get_model
from julearn.pipeline import PipelineCreator


def test_svm_calibrated_params() -> None:
    """Test the parameters of the calibrated SVC."""
    model = get_model(
        "svm_calibrated",
        problem_type="classification",
        method="isotonic",
        cv=3,
        C=3,
        estimator__kernel="linear",
    )
    assert isinstance(model, CalibratedClassifierCV)
    assert model.ensemble is False
    assert model.method == "isotonic"
    assert model.cv == 3
    assert isinstance(model.estimator, SVC)
    assert model.estimator.C == 3
    assert model.estimator.kernel == "linear"


def test_svm_calibrated_no_regression() -> None:
    """Test that the calibrated SVC is not available for regression."""
    with pytest.raises(ValueError, match="not suitable for regression"):
        get_model("svm_calibrated", problem_type="regression")


def test_svm_calibrated_tuning(df_iris: pd.DataFrame) -> None:
    """Test tuning the parameters of the calibrated SVC.

    Parameters
    ----------
    df_iris : pd.DataFrame
        The iris dataset.

    """
    X = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    creator = PipelineCreator(problem_type="classification")
    creator.add("zscore")
    creator.add("svm_calibrated", estimator__C=[0.01, 1, 100])
    _, estimator = run_cross_validation(  # type: ignore
        X=X,
        y="species",
        data=df_iris,
        model=creator,
        cv=3,
        scoring="accuracy",
        return_estimator="final",
    )
    assert list(estimator.param_grid) == [  # type: ignore
        "svm_calibrated__estimator__C"
    ]
    inner_scores = estimator.cv_results_["mean_test_score"]  # type: ignore
    assert len(set(inner_scores)) > 1


@pytest.mark.skipif(
    SVC().probability != "deprecated",
    reason="SVC probability is only deprecated in scikit-learn >= 1.9",
)
@pytest.mark.parametrize("probability", [True, [True, False]])
def test_svm_probability_warning(probability: bool | list[bool]) -> None:
    """Test the warning when using the deprecated SVC probability.

    Parameters
    ----------
    probability : bool or list of bool
        The value of the probability parameter.

    """
    with pytest.warns(FutureWarning, match="use the 'svm_calibrated' model"):
        PipelineCreator(problem_type="classification").add(
            "svm", probability=probability
        )


def test_svm_no_probability_warning() -> None:
    """Test that the SVC does not warn without probability."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        PipelineCreator(problem_type="classification").add(
            "svm", probability=False
        )
        PipelineCreator(problem_type="classification").add("svm", C=1)
