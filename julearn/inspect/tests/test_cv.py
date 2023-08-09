"""Provide tests for cross-validation inspection."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from sklearn.base import BaseEstimator
from sklearn.model_selection import RepeatedKFold

from julearn.base.estimators import WrapModel
from julearn.inspect import FoldsInspector, PipelineInspector
from julearn.pipeline import PipelineCreator
from julearn.utils import _compute_cvmdsum


class MockModelReturnsIndex(BaseEstimator):
    """Class for mock model."""

    def fit(self, X, y=None, **fit_params):  # noqa: N803
        """Fit the model."""
        return self

    def predict(self, X):  # noqa: N803
        """Predict using the model."""
        return np.array(X.index)[:, None]

    def predict_proba(self, X):  # noqa: N803
        """Predict probability using the model."""
        return np.array(X.index)[:, None]

    def predict_log_proba(self, X):  # noqa: N803
        """Predict log probability using the model."""
        return np.array(X.index)[:, None]

    def decision_function(self, X):  # noqa: N803
        """Decision function."""
        return np.array(X.index)[:, None]

    def __sklearn_is_fitted__(self):
        """Check if model is fitted on data."""
        return True


class MockRegressorReturnsIndex(BaseEstimator):
    """Class for mock regressor."""

    def fit(self, X, y=None, **fit_params):  # noqa: N803
        """Fit the model."""
        return self

    def predict(self, X):  # noqa: N803
        """Predict using the model."""
        return np.array(X.index)

    def __sklearn_is_fitted__(self):
        """Check if model is fitted on data."""
        return True


def scores(df_typed_iris, n_iters=5, mock_model=None):
    """Pre-define scores."""

    X = df_typed_iris.iloc[:, :-1]
    y = df_typed_iris.iloc[:, -1]

    if mock_model is None:
        mock_model = MockModelReturnsIndex

    estimators = [WrapModel(mock_model()).fit(X, y) for _ in range(n_iters)]

    return pd.DataFrame(
        {
            "estimator": estimators,
            "test_scores": [0.5] * n_iters,
            "repeat": 0,
            "fold": range(n_iters),
        }
    )


@pytest.fixture
def get_cv_scores(request, df_typed_iris):
    """Fixture for getting CV scores."""

    n_iters = request.param
    mock_model = None
    if isinstance(n_iters, list):
        n_iters, mock_model = n_iters

    cv = RepeatedKFold(n_repeats=1, n_splits=n_iters, random_state=2)
    cv_mdsum = _compute_cvmdsum(cv)
    df = scores(df_typed_iris, n_iters=n_iters, mock_model=mock_model)
    df["cv_mdsum"] = cv_mdsum
    return cv, df


@pytest.mark.parametrize("get_cv_scores", [2, 5, 10], indirect=True)
def test_get_predictions(get_cv_scores, df_typed_iris):
    """Test predictions."""

    X = df_typed_iris.iloc[:, :-1]
    y = df_typed_iris.iloc[:, -1]
    cv, df_scores = get_cv_scores
    inspector = FoldsInspector(df_scores, cv=cv, X=X, y=y)
    print(df_scores)
    expected_df = pd.DataFrame(
        {"repeat0_p0": X.index.values, "target": y.values}
    )
    assert_frame_equal(inspector.predict(), expected_df)
    assert_frame_equal(inspector.predict_proba(), expected_df)
    assert_frame_equal(inspector.predict_log_proba(), expected_df)
    assert_frame_equal(inspector.decision_function(), expected_df)


@pytest.mark.parametrize(
    "get_cv_scores", [[2, MockRegressorReturnsIndex]], indirect=True
)
def test_predictions_available(get_cv_scores, df_typed_iris):
    """Test available predictions."""

    X = df_typed_iris.iloc[:, :-1]
    y = df_typed_iris.iloc[:, -1]
    cv, df_scores = get_cv_scores
    inspector = FoldsInspector(df_scores, cv=cv, X=X, y=y)
    with pytest.raises(
        AttributeError,
        match="This 'FoldsInspector' has no attribute 'predict_proba'",
    ):
        inspector.predict_proba()

    with pytest.raises(
        AttributeError,
        match="This 'FoldsInspector' has no attribute 'predict_log_proba'",
    ):
        inspector.predict_log_proba()

    with pytest.raises(
        AttributeError,
        match="This 'FoldsInspector' has no attribute 'decision_function'",
    ):
        inspector.decision_function()


@pytest.mark.parametrize(
    "get_cv_scores", [[2, MockRegressorReturnsIndex]], indirect=True
)
def test_invalid_func(get_cv_scores, df_typed_iris):
    """Test invalid function."""

    X = df_typed_iris.iloc[:, :-1]
    y = df_typed_iris.iloc[:, -1]
    cv, df_scores = get_cv_scores
    inspector = FoldsInspector(df_scores, cv=cv, X=X, y=y)
    with pytest.raises(ValueError, match="Invalid func: no"):
        inspector._get_predictions("no")


@pytest.mark.parametrize("get_cv_scores", [5], indirect=True)
def test_foldsinspector_iter(get_cv_scores, df_typed_iris):
    """Test folds inspector iterations."""

    X = df_typed_iris.iloc[:, :-1]
    y = df_typed_iris.iloc[:, -1]
    cv, df_scores = get_cv_scores
    df_scores["estimator"] = [
        (
            PipelineCreator(problem_type="regression")
            .add(MockRegressorReturnsIndex())
            .to_pipeline()
            .fit(X, y)
        )
        for _ in range(len(df_scores))
    ]

    inspector = FoldsInspector(df_scores, cv=cv, X=X, y=y)

    for fold_inspector in inspector:
        i_model = fold_inspector.model

        assert isinstance(fold_inspector.model, PipelineInspector)
        assert isinstance(
            i_model.get_step("mockregressorreturnsindex").estimator,
            MockRegressorReturnsIndex,
        )
