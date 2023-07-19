from julearn.inspect import Inspector
from julearn import run_cross_validation, PipelineCreator
import pytest


def test_no_cv():
    inspector = Inspector(dict())
    with pytest.raises(ValueError, match="No cv"):
        inspector.folds


def test_no_X():
    inspector = Inspector(dict(), cv=5)
    with pytest.raises(ValueError, match="No X"):
        inspector.folds


def test_no_y():
    inspector = Inspector(dict(), cv=5, X=[1, 2, 3])
    with pytest.raises(ValueError, match="No y"):
        inspector.folds


def test_no_model():
    inspector = Inspector(dict())
    with pytest.raises(ValueError, match="No model"):
        inspector.model


def test_normal_usage(df_iris):
    X = list(df_iris.iloc[:, :-1].columns)
    scores, pipe, inspect = run_cross_validation(
        X=X, y="species", data=df_iris, model="svm",
        return_estimator="all", return_inspector=True,
        problem_type="classification"
    )
    assert pipe == inspect.model._model
    for (_, score), inspect_fold in zip(scores.iterrows(), inspect.folds):
        assert score["estimator"] == inspect_fold.model._model


def test_normal_usage_with_search(df_iris):
    X = list(df_iris.iloc[:, :-1].columns)

    pipe = PipelineCreator(problem_type="classification").add("svm", C=[1, 2])
    _, pipe, inspect = run_cross_validation(
        X=X, y="species", data=df_iris, model=pipe,
        return_estimator="all", return_inspector=True,
    )
    assert pipe == inspect.model._model
    inspect.model.get_fitted_params()
    inspect.model.get_params()
