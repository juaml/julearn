import pandas as pd
from sklearn.model_selection import RepeatedKFold
from sklearn.base import BaseEstimator
from julearn.base.estimators import WrapModel
from julearn.inspect import FoldsInspector
from julearn.utils import _compute_cvmdsum
from numpy.testing import assert_array_almost_equal
import pytest


class MockModelReturnsIndex(BaseEstimator):
    def fit(self, X, y=None, **fit_params):
        return self

    def predict(self, X):
        return X.index

    def predict_proba(self, X):
        return X.index

    def decision_function(self, X):
        return X.index


class MockRegressorReturnsIndex(BaseEstimator):
    def fit(self, X, y=None, **fit_params):
        return self

    def predict(self, X):
        return X.index


def scores(df_typed_iris, n_iters=5, mock_model=None):
    X = df_typed_iris.iloc[:, :-1]
    y = df_typed_iris.iloc[:, -1]

    if mock_model is None:
        mock_model = MockModelReturnsIndex

    estimators = [WrapModel(mock_model()).fit(X, y)
                  for _ in range(n_iters)]

    return pd.DataFrame(dict(
        estimator=estimators,
        test_scores=[.5]*n_iters,
        repeat=0, fold=range(n_iters)

    ))


@pytest.fixture
def get_cv_scores(request, df_typed_iris):
    n_iters = request.param
    mock_model = None
    if isinstance(n_iters, list):
        n_iters, mock_model = n_iters

    cv = RepeatedKFold(n_repeats=1, n_splits=n_iters, random_state=2)
    cv_mdsum = _compute_cvmdsum(cv)
    df = scores(df_typed_iris, n_iters=n_iters, mock_model=mock_model)
    df["cv_mdsum"] = cv_mdsum
    return cv, df


@pytest.mark.parametrize('get_cv_scores', [2, 5, 10], indirect=True)
def test_get_predictions(get_cv_scores, df_typed_iris):
    X = df_typed_iris.iloc[:, :-1]
    y = df_typed_iris.iloc[:, -1]
    cv, df_scores = get_cv_scores
    inspector = FoldsInspector(df_scores, cv=cv, X=X, y=y)
    assert_array_almost_equal(
        inspector.predict().values.flatten(),
        X.index.values
    )
    assert_array_almost_equal(
        inspector.predict_proba().values.flatten(),
        X.index.values
    )
    assert_array_almost_equal(
        inspector.decision_function().values.flatten(),
        X.index.values
    )


@pytest.mark.parametrize(
    'get_cv_scores', [
        [2, MockRegressorReturnsIndex]
    ], indirect=True)
def test_predictions_available(get_cv_scores, df_typed_iris):
    X = df_typed_iris.iloc[:, :-1]
    y = df_typed_iris.iloc[:, -1]
    cv, df_scores = get_cv_scores
    inspector = FoldsInspector(df_scores, cv=cv, X=X, y=y)
    with pytest.raises(
            AttributeError,
            match="This 'FoldsInspector' has no attribute 'predict_proba'"
    ):
        inspector.predict_proba()

    with pytest.raises(
            AttributeError,
            match="This 'FoldsInspector' has no attribute 'predict_log_proba'"
    ):
        inspector.predict_log_proba()

    with pytest.raises(
            AttributeError,
            match="This 'FoldsInspector' has no attribute 'decision_function'"
    ):
        inspector.decision_function()
