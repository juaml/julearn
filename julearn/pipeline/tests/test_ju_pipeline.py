"""Provide tests for the JuPipeline workaround."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
# License: AGPL

# TODO: Remove these tests together with julearn/pipeline/ju_pipeline.py once
# julearn only supports scikit-learn versions with the fix for
# https://github.com/scikit-learn/scikit-learn/issues/35195.

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import BaggingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from julearn import run_cross_validation
from julearn.pipeline import JuPipeline, PipelineCreator, ju_pipeline


@pytest.mark.parametrize(
    "estimator",
    [
        BaggingRegressor(LinearRegression()),
        VotingRegressor([("lr", LinearRegression())]),
    ],
)
def test_JuPipeline_score(
    X_iris: pd.DataFrame,  # noqa: N803
    estimator: BaggingRegressor | VotingRegressor,
) -> None:
    """Test that JuPipeline can be scored without sample_weight routing.

    The final step does not route sample_weight for score.

    Parameters
    ----------
    X_iris : pd.DataFrame
        The iris dataset features.
    estimator : BaggingRegressor or VotingRegressor
        The final step of the pipeline.

    """
    X = X_iris.iloc[:, :3]
    y = X_iris.iloc[:, 3]
    pipeline = JuPipeline([("model", estimator)]).fit(X, y)
    assert pipeline.score(X, y) == pytest.approx(estimator.score(X, y))


def test_PipelineCreator_uses_JuPipeline() -> None:
    """Test that PipelineCreator creates JuPipeline objects."""
    creator = PipelineCreator(problem_type="regression")
    creator.add("zscore").add("bagging", estimator=Ridge())
    pipeline = creator.to_pipeline()
    assert isinstance(pipeline, JuPipeline)
    assert isinstance(pipeline, Pipeline)


def test_run_cross_validation_default_scoring(df_iris: pd.DataFrame) -> None:
    """Test cross-validation with the default scoring and a bagging model.

    Parameters
    ----------
    df_iris : pd.DataFrame
        The iris dataset.

    """
    X = ["sepal_length", "sepal_width", "petal_length"]
    creator = PipelineCreator(problem_type="regression")
    creator.add("zscore").add("bagging", estimator=Ridge())
    scores = run_cross_validation(
        X=X, y="petal_width", data=df_iris, model=creator, cv=3
    )
    assert not np.any(np.isnan(scores["test_score"]))  # type: ignore


def test_tuning_default_scoring(df_iris: pd.DataFrame) -> None:
    """Test hyperparameter tuning with the default scoring of the search.

    Parameters
    ----------
    df_iris : pd.DataFrame
        The iris dataset.

    """
    X = ["sepal_length", "sepal_width", "petal_length"]
    creator = PipelineCreator(problem_type="regression")
    creator.add("zscore")
    creator.add("bagging", estimator=Ridge(), max_samples=[0.1, 1.0])
    _, estimator = run_cross_validation(  # type: ignore
        X=X,
        y="petal_width",
        data=df_iris,
        model=creator,
        cv=3,
        scoring="r2",
        return_estimator="final",
    )
    inner_scores = estimator.cv_results_["mean_test_score"]  # type: ignore
    assert not np.any(np.isnan(inner_scores))
    assert len(set(inner_scores)) == len(inner_scores)


def test_user_pipeline_cannot_score(df_iris: pd.DataFrame) -> None:
    """Test the error with a scikit-learn Pipeline that cannot be scored.

    Parameters
    ----------
    df_iris : pd.DataFrame
        The iris dataset.

    """
    X = ["sepal_length", "sepal_width", "petal_length"]
    model = GridSearchCV(
        make_pipeline(StandardScaler(), BaggingRegressor(Ridge())),
        {"baggingregressor__n_estimators": [2, 5]},
    )
    with pytest.raises(ValueError, match="ending in a BaggingRegressor"):
        run_cross_validation(
            X=X,
            y="petal_width",
            data=df_iris,
            model=model,
            problem_type="regression",
            cv=3,
        )


def test_user_pipeline_can_score(df_iris: pd.DataFrame) -> None:
    """Test that a scikit-learn Pipeline that can be scored is allowed.

    Parameters
    ----------
    df_iris : pd.DataFrame
        The iris dataset.

    """
    X = ["sepal_length", "sepal_width", "petal_length"]
    model = GridSearchCV(
        make_pipeline(StandardScaler(), Ridge()), {"ridge__alpha": [1, 10]}
    )
    scores = run_cross_validation(
        X=X,
        y="petal_width",
        data=df_iris,
        model=model,
        problem_type="regression",
        cv=3,
    )
    assert not np.any(np.isnan(scores["test_score"]))  # type: ignore


def test_check_pipelines_can_score_fixed_sklearn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that no error is raised if scikit-learn is fixed.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        The monkeypatch fixture.

    """
    monkeypatch.setattr(
        ju_pipeline, "_sklearn_pipeline_score_is_broken", lambda: False
    )
    model = make_pipeline(StandardScaler(), BaggingRegressor(Ridge()))
    ju_pipeline.check_pipelines_can_score(model)
