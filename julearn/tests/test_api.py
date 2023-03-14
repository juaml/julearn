"""Provide tests for the API."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL

import pandas as pd
import numpy as np

import pytest

from sklearn.svm import SVC
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import (
    StratifiedKFold,
    GridSearchCV,
    RepeatedKFold,
    cross_validate,
    RandomizedSearchCV,
)

from julearn import run_cross_validation
from julearn.utils.testing import do_scoring_test, compare_models
from julearn.pipeline import PipelineCreator


def test_run_cv_simple_binary(
    df_binary: pd.DataFrame, df_iris: pd.DataFrame
) -> None:
    """Test a simple binary classification problem.

    Parameters
    ----------
    df_binary : pd.DataFrame
        The iris dataset as a binary classification problem.
    df_iris : pd.DataFrame
        The iris dataset as a multiclass classification problem.
    """
    X = ["sepal_length", "sepal_width", "petal_length"]
    y = "species"

    scorers = ["accuracy", "balanced_accuracy"]
    api_params = {"model": "svm", "problem_type": "classification"}
    sklearn_model = SVC()

    with pytest.warns(RuntimeWarning, match="treated as continuous"):
        do_scoring_test(
            X=X,
            y=y,
            data=df_binary,
            scorers=scorers,
            api_params=api_params,
            sklearn_model=sklearn_model,
        )

    # now let"s try target-dependent scores
    scorers = ["recall", "precision", "f1"]
    sk_y = (df_iris[y].values == "virginica").astype(int)
    with pytest.warns(RuntimeWarning, match="treated as continuous"):
        api_params = {
            "model": "svm",
            "pos_labels": "virginica",
            "problem_type": "classification",
        }
        sklearn_model = SVC()
        do_scoring_test(
            X,
            y,
            data=df_iris,
            api_params=api_params,
            sklearn_model=sklearn_model,
            scorers=scorers,
            sk_y=sk_y,
        )

    # now let"s try proba-dependent scores
    X = ["sepal_length", "petal_length"]
    scorers = ["accuracy", "roc_auc"]
    sk_y = (df_iris[y].values == "virginica").astype(int)
    with pytest.warns(RuntimeWarning, match="treated as continuous"):
        api_params = {
            "model": "svm",
            "pos_labels": "virginica",
            "problem_type": "classification",
            "model_params": {"svm__probability": True},
        }
        sklearn_model = SVC(probability=True)
        do_scoring_test(
            X,
            y,
            data=df_iris,
            api_params=api_params,
            sklearn_model=sklearn_model,
            scorers=scorers,
            sk_y=sk_y,
        )

    # now let"s try for decision_function based scores
    # e.g. svm with probability=False
    X = ["sepal_length", "petal_length"]
    scorers = ["accuracy", "roc_auc"]
    sk_y = (df_iris[y].values == "virginica").astype(int)
    with pytest.warns(RuntimeWarning, match="treated as continuous"):
        api_params = {
            "model": "svm",
            "pos_labels": "virginica",
            "problem_type": "classification",
        }
        sklearn_model = SVC(probability=False)
        do_scoring_test(
            X,
            y,
            data=df_iris,
            api_params=api_params,
            sklearn_model=sklearn_model,
            scorers=scorers,
            sk_y=sk_y,
        )


def test_run_cv_simple_binary_errors(
    df_binary: pd.DataFrame, df_iris: pd.DataFrame
) -> None:
    """Test a simple classification problem errors

    Parameters
    ----------
    df_binary : pd.DataFrame
        The iris dataset as a binary classification problem.
    df_iris : pd.DataFrame
        The iris dataset as a multiclass classification problem.
    """

    # Test error when pos_labels are not provide (target-dependent scores)
    X = ["sepal_length", "sepal_width", "petal_length"]
    y = "species"
    scorers = ["recall", "precision", "f1"]
    api_params = {"model": "svm", "problem_type": "classification"}
    sklearn_model = SVC()

    with pytest.warns(UserWarning, match="Target is multiclass but average"):
        do_scoring_test(
            X,
            y,
            data=df_iris,
            api_params=api_params,
            sklearn_model=sklearn_model,
            scorers=scorers,
        )


def test_run_cv_errors(df_iris: pd.DataFrame) -> None:
    """Test a run_cross_validation errors and warnings.

    Parameters
    ----------
    df_iris : pd.DataFrame
        The iris dataset as a multiclass classification problem.
    """
    X = ["sepal_length", "sepal_width", "petal_length"]
    y = "species"

    X_types = {"continuous": X}
    # test error when model is a pipeline
    model = make_pipeline(SVC())
    with pytest.raises(ValueError, match="a scikit-learn pipeline"):
        run_cross_validation(
            X=X,
            y=y,
            data=df_iris,
            X_types=X_types,
            model=model,
            problem_type="classification",
        )

    # test error when model is a pipeline creator and preprocess is set
    model = PipelineCreator(problem_type="classification")
    with pytest.raises(ValueError, match="preprocess should be None"):
        run_cross_validation(
            X=X,
            y=y,
            data=df_iris,
            X_types=X_types,
            model=model,
            preprocess="zscore",
        )

    # test error when model is a pipeline creator and problem_type is set
    model = PipelineCreator(problem_type="classification")
    with pytest.raises(ValueError, match="Problem type should be set"):
        run_cross_validation(
            X=X,
            y=y,
            data=df_iris,
            X_types=X_types,
            model=model,
            problem_type="classification",
        )

    model = 2
    with pytest.raises(ValueError, match="has to be a PipelineCreator"):
        run_cross_validation(
            X=X,
            y=y,
            data=df_iris,
            X_types=X_types,
            model=model,
        )

    model = "svm"
    with pytest.raises(ValueError, match="`problem_type` must be specified"):
        run_cross_validation(
            X=X,
            y=y,
            data=df_iris,
            X_types=X_types,
            model=model,
        )

    model = "svm"
    with pytest.raises(ValueError, match="preprocess has to be a string"):
        run_cross_validation(
            X=X,
            y=y,
            data=df_iris,
            X_types=X_types,
            model=model,
            preprocess=2,
            problem_type="classification",
        )

    model = SVC()
    model_params = {"svc__C": 1}
    with pytest.raises(
        ValueError, match="Cannot use model_params with a model object"
    ):
        run_cross_validation(
            X=X,
            y=y,
            data=df_iris,
            X_types=X_types,
            model=model,
            model_params=model_params,
            problem_type="classification",
        )

    model = PipelineCreator(problem_type="classification")
    model_params = {"svc__C": 1}
    with pytest.raises(ValueError, match="must be None"):
        run_cross_validation(
            X=X,
            y=y,
            data=df_iris,
            X_types=X_types,
            model=model,
            model_params=model_params,
        )

    model = "svm"
    model_params = {"svc__C": 1}
    with pytest.raises(ValueError, match="model_params are incorrect"):
        run_cross_validation(
            X=X,
            y=y,
            data=df_iris,
            X_types=X_types,
            model=model,
            model_params=model_params,
            problem_type="classification",
        )

    model = "svm"
    model_params = {"probability": True, "svm__C": 1}
    with pytest.raises(ValueError, match="model_params are incorrect"):
        run_cross_validation(
            X=X,
            y=y,
            data=df_iris,
            X_types=X_types,
            model=model,
            model_params=model_params,
            problem_type="classification",
        )

    model = "svm"
    model_params = {"svm__C": 1}
    search_params = {"kind": "grid", "cv": 3}
    with pytest.raises(ValueError, match="earch parameters were specified"):
        run_cross_validation(
            X=X,
            y=y,
            data=df_iris,
            X_types=X_types,
            model=model,
            model_params=model_params,
            search_params=search_params,
            problem_type="classification",
        )


def test_tune_hyperparam_gridsearch(df_iris: pd.DataFrame) -> None:
    """Test a run_cross_validation with hyperparameter tunning (gridsearch).

    Parameters
    ----------
    df_iris : pd.DataFrame
        The iris dataset as a multiclass classification problem.
    """
    # keep only two species
    df_iris = df_iris[df_iris["species"].isin(["versicolor", "virginica"])]
    X = ["sepal_length", "sepal_width", "petal_length"]
    y = "species"
    X_types = {"continuous": X}

    sk_X = df_iris[X].values
    sk_y = df_iris[y].values

    scoring = "accuracy"

    np.random.seed(42)
    cv_outer = RepeatedKFold(n_splits=2, n_repeats=1)
    cv_inner = RepeatedKFold(n_splits=2, n_repeats=1)

    model_params = {"svm__C": [0.01, 0.001]}
    search_params = {"cv": cv_inner}
    actual, actual_estimator = run_cross_validation(
        X=X,
        y=y,
        data=df_iris,
        X_types=X_types,
        model="svm",
        model_params=model_params,
        cv=cv_outer,
        scoring=[scoring],
        return_estimator="final",
        search_params=search_params,
        problem_type="classification",
    )

    # Now do the same with scikit-learn
    np.random.seed(42)
    cv_outer = RepeatedKFold(n_splits=2, n_repeats=1)
    cv_inner = RepeatedKFold(n_splits=2, n_repeats=1)

    clf = make_pipeline(SVC())
    gs = GridSearchCV(clf, {"svc__C": [0.01, 0.001]}, cv=cv_inner)

    expected = cross_validate(gs, sk_X, sk_y, cv=cv_outer, scoring=[scoring])

    assert len(actual.columns) == len(expected) + 2
    assert len(actual["test_accuracy"]) == len(expected["test_accuracy"])
    assert all(
        [
            a == b
            for a, b in zip(actual["test_accuracy"], expected["test_accuracy"])
        ]
    )

    # Compare the models
    clf1 = actual_estimator.best_estimator_.steps[-1][1]
    clf2 = clone(gs).fit(sk_X, sk_y).best_estimator_.steps[-1][1]
    compare_models(clf1, clf2)


def test_tune_hyperparam_randomsearch(df_iris: pd.DataFrame) -> None:
    """Test a run_cross_validation with hyperparameter tunning (randomsearch).

    Parameters
    ----------
    df_iris : pd.DataFrame
        The iris dataset as a multiclass classification problem.
    """
    # keep only two species
    df_iris = df_iris[df_iris["species"].isin(["versicolor", "virginica"])]
    X = ["sepal_length", "sepal_width", "petal_length"]
    y = "species"
    X_types = {"continuous": X}

    sk_X = df_iris[X].values
    sk_y = df_iris[y].values

    scoring = "accuracy"

    # Now randomized search
    np.random.seed(42)
    cv_outer = RepeatedKFold(n_splits=2, n_repeats=1)
    cv_inner = RepeatedKFold(n_splits=2, n_repeats=1)
    model_params = {
        "svm__C": [0.01, 0.001],
    }
    search_params = {
        "kind": "random",
        "n_iter": 2,
        "cv": cv_inner,
    }
    actual, actual_estimator = run_cross_validation(
        X=X,
        y=y,
        data=df_iris,
        X_types=X_types,
        model="svm",
        model_params=model_params,
        search_params=search_params,
        problem_type="classification",
        cv=cv_outer,
        scoring=[scoring],
        return_estimator="final",
    )

    # Now do the same with scikit-learn
    np.random.seed(42)
    cv_outer = RepeatedKFold(n_splits=2, n_repeats=1)
    cv_inner = RepeatedKFold(n_splits=2, n_repeats=1)

    clf = make_pipeline(SVC())
    gs = RandomizedSearchCV(
        clf, {"svc__C": [0.01, 0.001]}, cv=cv_inner, n_iter=2
    )

    expected = cross_validate(gs, sk_X, sk_y, cv=cv_outer, scoring=[scoring])

    assert len(actual.columns) == len(expected) + 2
    assert len(actual["test_accuracy"]) == len(expected["test_accuracy"])
    assert all(
        [
            a == b
            for a, b in zip(actual["test_accuracy"], expected["test_accuracy"])
        ]
    )

    # Compare the models
    clf1 = actual_estimator.best_estimator_.steps[-1][1]
    clf2 = clone(gs).fit(sk_X, sk_y).best_estimator_.steps[-1][1]
    compare_models(clf1, clf2)


def test_return_estimators(df_iris: pd.DataFrame) -> None:
    """Test returning estimators.

    Parameters
    ----------
    df_iris : pd.DataFrame
        The iris dataset as a multiclass classification problem.
    """
    df_iris = df_iris[df_iris["species"].isin(["versicolor", "virginica"])]
    X = ["sepal_length", "sepal_width", "petal_length"]
    y = "species"
    X_types = {"continuous": X}

    cv = StratifiedKFold(2)

    scores = run_cross_validation(
        X=X,
        y=y,
        data=df_iris,
        X_types=X_types,
        model="svm",
        problem_type="classification",
        cv=cv,
        return_estimator=None,
    )

    assert isinstance(scores, pd.DataFrame)
    assert "estimator" not in scores

    scores, final = run_cross_validation(
        X=X,
        y=y,
        data=df_iris,
        X_types=X_types,
        model="svm",
        problem_type="classification",
        cv=cv,
        return_estimator="final",
    )

    assert isinstance(scores, pd.DataFrame)
    assert "estimator" not in scores
    assert isinstance(final["svm"], SVC)

    scores = run_cross_validation(
        X=X,
        y=y,
        data=df_iris,
        X_types=X_types,
        model="svm",
        problem_type="classification",
        cv=cv,
        return_estimator="cv",
    )

    assert isinstance(scores, pd.DataFrame)
    assert "estimator" in scores

    scores, final = run_cross_validation(
        X=X,
        y=y,
        data=df_iris,
        X_types=X_types,
        model="svm",
        problem_type="classification",
        cv=cv,
        return_estimator="all",
    )

    assert isinstance(scores, pd.DataFrame)
    assert "estimator" in scores
    assert isinstance(final["svm"], SVC)


def test_return_train_scores(df_iris: pd.DataFrame) -> None:
    """Test returning estimators.

    Parameters
    ----------
    df_iris : pd.DataFrame
        The iris dataset as a multiclass classification problem.
    """
    df_iris = df_iris[df_iris["species"].isin(["versicolor", "virginica"])]
    X = ["sepal_length", "sepal_width", "petal_length"]
    y = "species"

    scoring = ["accuracy", "precision", "recall"]
    cv = StratifiedKFold(2)

    with pytest.warns(RuntimeWarning, match="treated as continuous"):
        scores = run_cross_validation(
            X=X,
            y=y,
            data=df_iris,
            model="svm",
            problem_type="classification",
            cv=cv,
            scoring=scoring,
        )

    train_scores = [f"train_{s}" for s in scoring]
    test_scores = [f"test_{s}" for s in scoring]

    assert all([s not in scores.columns for s in train_scores])
    assert all([s in scores.columns for s in test_scores])

    with pytest.warns(RuntimeWarning, match="treated as continuous"):
        scores = run_cross_validation(
            X=X,
            y=y,
            data=df_iris,
            model="svm",
            problem_type="classification",
            cv=cv,
            scoring=scoring,
            return_train_score=True,
        )

    train_scores = [f"train_{s}" for s in scoring]
    test_scores = [f"test_{s}" for s in scoring]

    assert all([s in scores.columns for s in train_scores])
    assert all([s in scores.columns for s in test_scores])
