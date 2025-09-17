"""Provide tests for cross-validation inspection."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from sklearn.base import BaseEstimator
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    KFold,
    LeaveOneGroupOut,
    LeaveOneOut,
    LeavePGroupsOut,
    LeavePOut,
    PredefinedSplit,
    RepeatedKFold,
    RepeatedStratifiedKFold,
    ShuffleSplit,
    StratifiedGroupKFold,
    StratifiedKFold,
    StratifiedShuffleSplit,
)

from julearn.base.estimators import WrapModel
from julearn.inspect import FoldsInspector, PipelineInspector
from julearn.model_selection import (
    ContinuousStratifiedGroupKFold,
    ContinuousStratifiedKFold,
    RepeatedContinuousStratifiedGroupKFold,
    RepeatedContinuousStratifiedKFold,
    StratifiedBootstrap,
)
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


def scores(
    df_typed_iris, n_splits=5, n_repeats=1, mock_model=None, target="species"
):
    """Pre-define scores."""

    X = df_typed_iris.iloc[:, :4]
    y = df_typed_iris[target]

    if mock_model is None:
        mock_model = MockModelReturnsIndex

    estimators = [
        WrapModel(mock_model()).fit(X, y)  # type: ignore
        for _ in range(n_splits * n_repeats)
    ]

    return pd.DataFrame(
        {
            "estimator": estimators,
            "test_scores": [0.5] * (n_splits * n_repeats),
            "fold": list(range(n_splits)) * n_repeats,
            "repeat": [i for i in range(n_repeats) for _ in range(n_splits)],
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
    df = scores(
        df_typed_iris, n_splits=n_iters, n_repeats=1, mock_model=mock_model
    )
    df["cv_mdsum"] = cv_mdsum
    return cv, df


@pytest.mark.parametrize("get_cv_scores", [2, 5, 10], indirect=True)
def test_get_predictions(get_cv_scores, df_typed_iris):
    """Test predictions."""

    X = df_typed_iris.iloc[:, :4]
    y = df_typed_iris["species"]
    # Add another kind of index to y
    y.index = [f"sample-{i:03d}" for i in range(100, len(y) + 100)]
    y.index.name = "sample_id"
    cv, df_scores = get_cv_scores
    inspector = FoldsInspector(df_scores, cv=cv, X=X, y=y)
    print(df_scores)
    expected_df = pd.DataFrame(
        {
            "sample_id": y.index.values,
            "target": y.values,
            "repeat0_p0": X.index.values,
        }
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


@pytest.mark.parametrize(
    "klass,params,cv_params",
    [
        [ShuffleSplit, [], {"n_splits": 5, "random_state": 2}],
        [GroupShuffleSplit, ["groups"], {"n_splits": 5, "random_state": 2}],
        [PredefinedSplit, [], {"test_fold": [1, 2, 3, 4, 5]}],
        [StratifiedShuffleSplit, [], {"n_splits": 5, "random_state": 2}],
        [StratifiedBootstrap, [], {"n_splits": 5, "random_state": 2}],
        [LeavePGroupsOut, ["groups"], {"n_groups": 1}],
        [LeavePOut, [], {"p": 2}],
    ],
)
def test_overlapping_cv_predictions(klass, params, cv_params, df_grouped_iris):
    """Test overlapping CV predictions."""
    mock_model = MockModelReturnsIndex
    X = df_grouped_iris.iloc[:, :4]
    y = df_grouped_iris["species"]
    # Add another kind of index to y
    y.index = [f"sample-{i:03d}" for i in range(100, len(y) + 100)]
    y.index.name = "sample_id"
    groups = None
    if "groups" in params:
        groups = df_grouped_iris["group"]

    cv = klass(**cv_params)
    cv_mdsum = _compute_cvmdsum(cv)
    n_repeats = getattr(cv, "n_repeats", 1)
    n_splits = cv.get_n_splits(X, y, groups) // n_repeats
    df_scores = scores(
        df_grouped_iris,
        n_splits=n_splits,
        n_repeats=n_repeats,
        mock_model=mock_model,
    )
    df_scores["cv_mdsum"] = cv_mdsum

    inspector = FoldsInspector(df_scores, cv=cv, X=X, y=y, groups=groups)
    print(df_scores)
    expected_dict = {}
    expected_dict["sample_id"] = y.index.values
    expected_dict["target"] = y.values
    for i, (_, test) in enumerate(cv.split(X, y, groups=groups)):
        expected_dict[f"fold{i}_p0"] = np.array(
            [j if j in test else np.nan for j in range(len(X))]
        )
    expected_df = pd.DataFrame(expected_dict)
    assert_frame_equal(inspector.predict(), expected_df)
    assert_frame_equal(inspector.predict_proba(), expected_df)
    assert_frame_equal(inspector.predict_log_proba(), expected_df)
    assert_frame_equal(inspector.decision_function(), expected_df)


@pytest.mark.parametrize(
    "klass,params,cv_params",
    [
        [KFold, [], {"n_splits": 5, "shuffle": True, "random_state": 2}],
        [
            GroupKFold,
            ["groups"],
            {"n_splits": 5, "shuffle": True, "random_state": 2},
        ],
        [
            RepeatedKFold,
            [],
            {"n_splits": 5, "n_repeats": 2, "random_state": 2},
        ],
        [LeaveOneGroupOut, ["groups"], {}],
        [LeaveOneOut, [], {}],
        [
            StratifiedKFold,
            [],
            {"n_splits": 5, "shuffle": True, "random_state": 2},
        ],
        [
            RepeatedStratifiedKFold,
            [],
            {"n_splits": 5, "n_repeats": 2, "random_state": 2},
        ],
        [
            StratifiedGroupKFold,
            ["groups"],
            {"n_splits": 5, "shuffle": True, "random_state": 2},
        ],
    ],
)
def test_nonoverlapping_cv_predictions(
    klass, params, cv_params, df_grouped_iris
):
    """Test non-overlapping CV predictions."""
    mock_model = MockModelReturnsIndex
    X = df_grouped_iris.iloc[:, :4]
    y = df_grouped_iris["species"]
    # Add another kind of index to y
    y.index = [f"sample-{i:03d}" for i in range(100, len(y) + 100)]
    y.index.name = "sample_id"
    groups = None
    if "groups" in params:
        groups = df_grouped_iris["group"]

    cv = klass(**cv_params)
    cv_mdsum = _compute_cvmdsum(cv)
    n_repeats = getattr(cv, "n_repeats", 1)
    n_splits = cv.get_n_splits(X, y, groups) // n_repeats
    df_scores = scores(
        df_grouped_iris,
        n_splits=n_splits,
        n_repeats=n_repeats,
        mock_model=mock_model,
    )
    df_scores["cv_mdsum"] = cv_mdsum

    inspector = FoldsInspector(df_scores, cv=cv, X=X, y=y, groups=groups)
    print(df_scores)
    expected_dict = {}
    expected_dict["sample_id"] = y.index.values
    expected_dict["target"] = y.values
    n_repeats = getattr(cv, "n_repeats", 1)
    for i in range(n_repeats):
        expected_dict[f"repeat{i}_p0"] = X.index.values
    expected_df = pd.DataFrame(expected_dict)
    assert_frame_equal(inspector.predict(), expected_df)
    assert_frame_equal(inspector.predict_proba(), expected_df)
    assert_frame_equal(inspector.predict_log_proba(), expected_df)
    assert_frame_equal(inspector.decision_function(), expected_df)


@pytest.mark.parametrize(
    "klass,params,cv_params",
    [
        [
            ContinuousStratifiedKFold,
            [],
            {"n_bins": 10, "n_splits": 5, "shuffle": True, "random_state": 2},
        ],
        [
            RepeatedContinuousStratifiedKFold,
            [],
            {
                "n_bins": 10,
                "n_splits": 5,
                "n_repeats": 3,
                "random_state": 2,
            },
        ],
        [
            ContinuousStratifiedGroupKFold,
            ["groups"],
            {"n_bins": 10, "n_splits": 5, "shuffle": True, "random_state": 2},
        ],
        [
            RepeatedContinuousStratifiedGroupKFold,
            ["groups"],
            {
                "n_bins": 10,
                "n_splits": 5,
                "n_repeats": 3,
                "random_state": 2,
            },
        ],
    ],
)
def test_nonoverlapping_continuous_cv_predictions(
    klass, params, cv_params, df_regression
):
    """Test non-overlapping continuouos CV predictions."""
    mock_model = MockModelReturnsIndex
    X = df_regression.iloc[:, :-1]
    y = df_regression["target"]
    # Add another kind of index to y
    y.index = [f"sample-{i:03d}" for i in range(100, len(y) + 100)]
    y.index.name = "sample_id"
    groups = None
    if "groups" in params:
        groups = df_regression["group"]

    cv = klass(**cv_params)
    cv_mdsum = _compute_cvmdsum(cv)
    n_repeats = getattr(cv, "n_repeats", 1)
    n_splits = cv.get_n_splits(X, y, groups) // n_repeats
    df_scores = scores(
        df_regression,
        n_splits=n_splits,
        n_repeats=n_repeats,
        mock_model=mock_model,
        target="target",
    )
    df_scores["cv_mdsum"] = cv_mdsum

    inspector = FoldsInspector(df_scores, cv=cv, X=X, y=y, groups=groups)
    print(df_scores)
    expected_dict = {}
    expected_dict["sample_id"] = y.index.values
    expected_dict["target"] = y.values
    n_repeats = getattr(cv, "n_repeats", 1)
    for i in range(n_repeats):
        expected_dict[f"repeat{i}_p0"] = X.index.values
    expected_df = pd.DataFrame(expected_dict)
    assert_frame_equal(inspector.predict(), expected_df)
    assert_frame_equal(inspector.predict_proba(), expected_df)
    assert_frame_equal(inspector.predict_log_proba(), expected_df)
    assert_frame_equal(inspector.decision_function(), expected_df)
