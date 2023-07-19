"""Provide tests for the API."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL

import numpy as np
import pandas as pd
import joblib
import pytest
from sklearn.base import clone
from sklearn.datasets import make_regression
from sklearn.model_selection import (
    GridSearchCV,
    GroupKFold,
    GroupShuffleSplit,
    KFold,
    LeaveOneGroupOut,
    LeaveOneOut,
    LeavePGroupsOut,
    LeavePOut,
    PredefinedSplit,
    RandomizedSearchCV,
    RepeatedKFold,
    RepeatedStratifiedKFold,
    ShuffleSplit,
    StratifiedGroupKFold,
    StratifiedKFold,
    StratifiedShuffleSplit,
    check_cv,
    cross_validate,
)
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

from julearn import run_cross_validation
from julearn.api import _compute_cvmdsum
from julearn.model_selection import (
    RepeatedContinuousStratifiedGroupKFold,
    ContinuousStratifiedGroupKFold,
)
from julearn.pipeline import PipelineCreator
from julearn.utils.testing import compare_models, do_scoring_test


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
    X_types = {"features": X}

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

    model = PipelineCreator(apply_to="features", problem_type="classification")
    model.add("svm")

    api_params = {
        "model": model,
        "pos_labels": "virginica",
    }
    sklearn_model = SVC()
    do_scoring_test(
        X,
        y,
        data=df_iris,
        api_params=api_params,
        X_types=X_types,
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


def test_run_cv_simple_binary_groups(df_iris: pd.DataFrame) -> None:
    """Test a simple binary classification problem with groups in the CV.

    Parameters
    ----------
    df_iris : pd.DataFrame
        The iris dataset as a multiclass classification problem.
    """
    # keep only two species
    df_iris = df_iris[df_iris["species"].isin(["versicolor", "virginica"])]
    df_iris = df_iris.copy()

    X = ["sepal_length", "sepal_width", "petal_length"]
    y = "species"
    X_types = {"continuous": X}

    df_iris["groups"] = np.digitize(
        df_iris["sepal_length"],
        bins=np.histogram(df_iris["sepal_length"], bins=20)[1],
    )

    scorers = ["accuracy", "balanced_accuracy"]
    api_params = {
        "model": "svm",
        "problem_type": "classification",
    }
    sklearn_model = SVC()
    cv = GroupKFold(n_splits=2)

    do_scoring_test(
        X=X,
        y=y,
        data=df_iris,
        X_types=X_types,
        groups="groups",
        cv=cv,
        scorers=scorers,
        api_params=api_params,
        sklearn_model=sklearn_model,
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


def test_run_cv_multiple_pipeline_errors(df_iris: pd.DataFrame) -> None:
    """Test run_cross_validation with multiple pipelines errors."""
    X = ["sepal_length", "sepal_width", "petal_length"]
    y = "species"

    X_types = {"continuous": X}
    model1 = PipelineCreator(problem_type="classification")
    model1.add("svm")
    model2 = "svm"
    with pytest.raises(ValueError, match="If model is a list, all"):
        run_cross_validation(
            X=X,
            y=y,
            data=df_iris,
            X_types=X_types,
            model=[model1, model2],  # type: ignore
        )

    model2 = PipelineCreator(problem_type="regression")
    model2.add("svm")

    with pytest.raises(ValueError, match="same problem_type"):
        run_cross_validation(
            X=X,
            y=y,
            data=df_iris,
            X_types=X_types,
            model=[model1, model2],  # type: ignore
        )


def test_tune_hyperparam_gridsearch(df_iris: pd.DataFrame) -> None:
    """Test a run_cross_validation with hyperparameter tuning (gridsearch).

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

    assert len(actual.columns) == len(expected) + 5
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


def test_tune_hyperparam_gridsearch_groups(df_iris: pd.DataFrame) -> None:
    """Test a run_cross_validation with hyperparameter tuning (gridsearch).

    Parameters
    ----------
    df_iris : pd.DataFrame
        The iris dataset as a multiclass classification problem.
    """
    # keep only two species
    df_iris = df_iris[df_iris["species"].isin(["versicolor", "virginica"])]
    df_iris = df_iris.copy()
    X = ["sepal_length", "sepal_width", "petal_length"]
    y = "species"
    X_types = {"continuous": X}

    df_iris["groups"] = np.digitize(
        df_iris["sepal_length"],
        bins=np.histogram(df_iris["sepal_length"], bins=20)[1],
    )

    sk_X = df_iris[X].values
    sk_y = df_iris[y].values
    sk_groups = df_iris["groups"].values

    scoring = "accuracy"

    np.random.seed(42)
    cv_outer = GroupKFold(n_splits=2)
    cv_inner = GroupKFold(n_splits=2)

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
        groups="groups",
        return_estimator="final",
        search_params=search_params,
        problem_type="classification",
    )

    # Now do the same with scikit-learn
    np.random.seed(42)
    cv_outer = GroupKFold(n_splits=2)
    cv_inner = GroupKFold(n_splits=2)

    clf = make_pipeline(SVC())
    gs = GridSearchCV(clf, {"svc__C": [0.01, 0.001]}, cv=cv_inner)

    expected = cross_validate(
        gs,
        sk_X,
        sk_y,
        cv=cv_outer,
        scoring=[scoring],
        groups=sk_groups,
        fit_params={"groups": sk_groups},
    )

    assert len(actual.columns) == len(expected) + 5
    assert len(actual["test_accuracy"]) == len(expected["test_accuracy"])
    assert all(
        [
            a == b
            for a, b in zip(actual["test_accuracy"], expected["test_accuracy"])
        ]
    )

    # Compare the models
    clf1 = actual_estimator.best_estimator_.steps[-1][1]
    clf2 = (
        clone(gs)
        .fit(sk_X, sk_y, groups=sk_groups)
        .best_estimator_.steps[-1][1]
    )
    compare_models(clf1, clf2)


def test_tune_hyperparam_randomsearch(df_iris: pd.DataFrame) -> None:
    """Test a run_cross_validation with hyperparameter tuning (randomsearch).

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

    assert len(actual.columns) == len(expected) + 5
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


def test_tune_hyperparams_multiple_grid(df_iris: pd.DataFrame) -> None:
    """Test a run_cross_validation hyperparameter tuning (multiple grid)."""

    df_iris = df_iris[df_iris["species"].isin(["versicolor", "virginica"])]
    X = ["sepal_length", "sepal_width", "petal_length"]
    y = "species"
    X_types = {"continuous": X}

    # Use a single creator with a repeated step name
    creator1 = PipelineCreator(problem_type="classification")
    creator1.add("svm", kernel="linear", C=[0.01, 0.1], name="svm")
    creator1.add(
        "svm",
        kernel="rbf",
        C=[0.01, 0.1],
        gamma=["scale", "auto", 1e-2, 1e-3],
        name="svm",
    )

    sk_X = df_iris[X].values
    sk_y = df_iris[y].values

    scoring = "accuracy"

    np.random.seed(42)
    cv_outer = RepeatedKFold(n_splits=2, n_repeats=1)
    cv_inner = RepeatedKFold(n_splits=2, n_repeats=1)

    search_params = {"cv": cv_inner}
    actual1, actual_estimator1 = run_cross_validation(
        X=X,
        y=y,
        data=df_iris,
        X_types=X_types,
        model=creator1,
        cv=cv_outer,
        scoring=[scoring],
        return_estimator="final",
        search_params=search_params,
    )

    # Use two creators
    creator2_1 = PipelineCreator(problem_type="classification")
    creator2_1.add("svm", kernel="linear", C=[0.01, 0.1], name="svm")
    creator2_2 = PipelineCreator(problem_type="classification")
    creator2_2.add(
        "svm",
        kernel="rbf",
        C=[0.01, 0.1],
        gamma=["scale", "auto", 1e-2, 1e-3],
        name="svm",
    )

    np.random.seed(42)
    cv_outer = RepeatedKFold(n_splits=2, n_repeats=1)
    cv_inner = RepeatedKFold(n_splits=2, n_repeats=1)
    search_params = {"cv": cv_inner}
    actual2, actual_estimator2 = run_cross_validation(
        X=X,
        y=y,
        data=df_iris,
        X_types=X_types,
        model=[creator2_1, creator2_2],
        cv=cv_outer,
        scoring=[scoring],
        return_estimator="final",
        search_params=search_params,
    )

    # Now do the same with scikit-learn
    np.random.seed(42)
    cv_outer = RepeatedKFold(n_splits=2, n_repeats=1)
    cv_inner = RepeatedKFold(n_splits=2, n_repeats=1)

    clf = make_pipeline(SVC())
    grid = [
        {
            "svc__C": [0.01, 0.1],
            "svc__kernel": ["linear"],
        },
        {
            "svc__gamma": ["scale", "auto", 1e-2, 1e-3],
            "svc__kernel": ["rbf"],
            "svc__C": [0.01, 0.1],
        },
    ]
    gs = GridSearchCV(clf, grid, cv=cv_inner)

    expected = cross_validate(gs, sk_X, sk_y, cv=cv_outer, scoring=[scoring])

    assert len(actual1.columns) == len(expected) + 5
    assert len(actual2.columns) == len(expected) + 5
    assert len(actual1["test_accuracy"]) == len(expected["test_accuracy"])
    assert len(actual2["test_accuracy"]) == len(expected["test_accuracy"])
    assert all(
        [
            a == b
            for a, b in zip(
                actual1["test_accuracy"], expected["test_accuracy"]
            )
        ]
    )
    assert all(
        [
            a == b
            for a, b in zip(
                actual2["test_accuracy"], expected["test_accuracy"]
            )
        ]
    )
    # Compare the models
    clf1 = actual_estimator1.best_estimator_.steps[-1][1]
    clf2 = actual_estimator2.best_estimator_.steps[-1][1]
    clf3 = clone(gs).fit(sk_X, sk_y).best_estimator_.steps[-1][1]
    compare_models(clf1, clf2)
    compare_models(clf1, clf3)


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

    with pytest.raises(ValueError, match="must be one of"):
        scores = run_cross_validation(
            X=X,
            y=y,
            data=df_iris,
            X_types=X_types,
            model="svm",
            problem_type="classification",
            cv=cv,
            return_estimator=True,
        )

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


@pytest.mark.parametrize(
    "cv1, cv2, expected",
    [
        (GroupKFold(2), KFold(3), False),
        (GroupKFold(2), GroupKFold(3), False),
        (GroupKFold(3), GroupKFold(3), True),
        (GroupShuffleSplit(2), GroupShuffleSplit(3), "non-reproducible"),
        (
            GroupShuffleSplit(2, random_state=32),
            GroupShuffleSplit(3, random_state=32),
            False,
        ),
        (
            GroupShuffleSplit(3, random_state=32),
            GroupShuffleSplit(3, random_state=32),
            True,
        ),
        (
            GroupShuffleSplit(3, random_state=33),
            GroupShuffleSplit(3, random_state=32),
            False,
        ),
        (KFold(2), KFold(3), False),
        (
            KFold(2, shuffle=True),
            KFold(2, shuffle=True),
            "non-reproducible",
        ),
        (
            KFold(3, random_state=32, shuffle=True),
            KFold(3, random_state=32, shuffle=True),
            True,
        ),
        (
            KFold(3, random_state=33, shuffle=True),
            KFold(3, random_state=32, shuffle=True),
            False,
        ),
        (LeaveOneGroupOut(), LeaveOneGroupOut(), True),
        (LeavePGroupsOut(3), LeavePGroupsOut(3), True),
        (LeavePGroupsOut(3), LeavePGroupsOut(2), False),
        (LeaveOneOut(), LeaveOneOut(), True),
        (LeavePOut(2), LeavePOut(2), True),
        (LeavePOut(2), LeavePOut(3), False),
        (PredefinedSplit([1, 2, 3]), PredefinedSplit([1, 2, 3]), True),
        (PredefinedSplit([1, 2, 3]), PredefinedSplit([1, 2, 4]), False),
        (
            RepeatedKFold(n_splits=2),
            RepeatedKFold(n_splits=2),
            "non-reproducible",
        ),
        (
            RepeatedKFold(n_splits=2, random_state=32),
            RepeatedKFold(n_splits=3, random_state=32),
            False,
        ),
        (
            RepeatedKFold(n_splits=2, random_state=32),
            RepeatedKFold(n_splits=2, random_state=32),
            True,
        ),
        (
            RepeatedKFold(n_splits=2, n_repeats=2, random_state=32),
            RepeatedKFold(n_splits=2, n_repeats=3, random_state=32),
            False,
        ),
        (
            RepeatedStratifiedKFold(n_splits=2),
            RepeatedStratifiedKFold(n_splits=2),
            "non-reproducible",
        ),
        (
            RepeatedStratifiedKFold(n_splits=2, random_state=32),
            RepeatedStratifiedKFold(n_splits=3, random_state=32),
            False,
        ),
        (
            RepeatedStratifiedKFold(n_splits=2, random_state=32),
            RepeatedStratifiedKFold(n_splits=2, random_state=32),
            True,
        ),
        (
            RepeatedStratifiedKFold(n_splits=2, n_repeats=2, random_state=32),
            RepeatedStratifiedKFold(n_splits=2, n_repeats=3, random_state=32),
            False,
        ),
        (
            ShuffleSplit(n_splits=2),
            ShuffleSplit(n_splits=2),
            "non-reproducible",
        ),
        (
            ShuffleSplit(n_splits=2, random_state=32),
            ShuffleSplit(n_splits=3, random_state=32),
            False,
        ),
        (
            ShuffleSplit(n_splits=2, random_state=32),
            ShuffleSplit(n_splits=2, random_state=32),
            True,
        ),
        (
            ShuffleSplit(n_splits=2, test_size=2, random_state=32),
            ShuffleSplit(n_splits=2, test_size=3, random_state=32),
            False,
        ),
        (
            ShuffleSplit(n_splits=2, train_size=2, random_state=32),
            ShuffleSplit(n_splits=2, train_size=3, random_state=32),
            False,
        ),
        (StratifiedKFold(2), StratifiedKFold(3), False),
        (
            StratifiedKFold(2, shuffle=True),
            StratifiedKFold(2, shuffle=True),
            "non-reproducible",
        ),
        (
            StratifiedKFold(3, random_state=32, shuffle=True),
            StratifiedKFold(3, random_state=32, shuffle=True),
            True,
        ),
        (
            StratifiedKFold(3, random_state=33, shuffle=True),
            StratifiedKFold(3, random_state=32, shuffle=True),
            False,
        ),
        (
            StratifiedShuffleSplit(n_splits=2),
            StratifiedShuffleSplit(n_splits=2),
            "non-reproducible",
        ),
        (
            StratifiedShuffleSplit(n_splits=2, random_state=32),
            StratifiedShuffleSplit(n_splits=3, random_state=32),
            False,
        ),
        (
            StratifiedShuffleSplit(n_splits=2, random_state=32),
            StratifiedShuffleSplit(n_splits=2, random_state=32),
            True,
        ),
        (
            StratifiedShuffleSplit(n_splits=2, test_size=2, random_state=32),
            StratifiedShuffleSplit(n_splits=2, test_size=3, random_state=32),
            False,
        ),
        (
            StratifiedShuffleSplit(n_splits=2, train_size=2, random_state=32),
            StratifiedShuffleSplit(n_splits=2, train_size=3, random_state=32),
            False,
        ),
        (StratifiedGroupKFold(2), StratifiedGroupKFold(3), False),
        (StratifiedGroupKFold(3), StratifiedGroupKFold(3), True),
        (
            ContinuousStratifiedGroupKFold(n_bins=10, n_splits=2),
            ContinuousStratifiedGroupKFold(n_bins=10, n_splits=3),
            False,
        ),
        (
            ContinuousStratifiedGroupKFold(n_bins=10, n_splits=2),
            ContinuousStratifiedGroupKFold(n_bins=11, n_splits=2),
            False,
        ),
        (
            ContinuousStratifiedGroupKFold(
                n_bins=10, n_splits=2, method="quantile"
            ),
            ContinuousStratifiedGroupKFold(n_bins=10, n_splits=2),
            False,
        ),
        (
            ContinuousStratifiedGroupKFold(
                n_bins=10, n_splits=2, shuffle=True
            ),
            ContinuousStratifiedGroupKFold(
                n_bins=10, n_splits=2, shuffle=True
            ),
            "non-reproducible",
        ),
        (
            ContinuousStratifiedGroupKFold(
                n_bins=10, n_splits=3, random_state=32, shuffle=True
            ),
            ContinuousStratifiedGroupKFold(
                n_bins=10, n_splits=3, random_state=32, shuffle=True
            ),
            True,
        ),
        (
            ContinuousStratifiedGroupKFold(
                n_bins=10, n_splits=3, random_state=33, shuffle=True
            ),
            ContinuousStratifiedGroupKFold(
                n_bins=10, n_splits=3, random_state=32, shuffle=True
            ),
            False,
        ),
        (
            RepeatedContinuousStratifiedGroupKFold(n_bins=10, n_splits=2),
            RepeatedContinuousStratifiedGroupKFold(n_bins=10, n_splits=2),
            "non-reproducible",
        ),
        (
            RepeatedContinuousStratifiedGroupKFold(
                n_bins=10, n_splits=2, random_state=32
            ),
            RepeatedContinuousStratifiedGroupKFold(
                n_bins=10, n_splits=3, random_state=32
            ),
            False,
        ),
        (
            RepeatedContinuousStratifiedGroupKFold(
                n_bins=10, n_splits=2, random_state=32
            ),
            RepeatedContinuousStratifiedGroupKFold(
                n_bins=10, n_splits=2, random_state=32
            ),
            True,
        ),
        (
            RepeatedContinuousStratifiedGroupKFold(
                n_bins=10, n_splits=2, n_repeats=2, random_state=32
            ),
            RepeatedContinuousStratifiedGroupKFold(
                n_bins=10, n_splits=2, n_repeats=3, random_state=32
            ),
            False,
        ),
        (
            [
                (np.arange(2, 9), np.arange(0, 2)),
                (np.arange(0, 7), np.arange(7, 9)),
            ],
            [
                (np.arange(2, 9), np.arange(0, 2)),
                (np.arange(0, 7), np.arange(7, 9)),
            ],
            True,
        ),
        (
            [
                (np.arange(3, 9), np.arange(0, 3)),
                (np.arange(0, 7), np.arange(7, 9)),
            ],
            [
                (np.arange(2, 9), np.arange(0, 2)),
                (np.arange(0, 7), np.arange(7, 9)),
            ],
            False,
        ),
    ],
)
def test__compute_cvmdsum(cv1, cv2, expected):
    """Test _compute_cvmdsum."""
    cv1 = check_cv(cv1)
    cv2 = check_cv(cv2)
    md1 = _compute_cvmdsum(cv1)
    md2 = _compute_cvmdsum(cv2)
    if expected == "non-reproducible":
        assert md1 == md2
        assert md1 == expected
    else:
        assert (md1 == md2) is expected


def test_api_stacking_models() -> None:
    """ "Test API of stacking models."""
    # prepare data
    X, y = make_regression(n_features=6, n_samples=50)

    # prepare feature names and types
    X_types = {
        "type1": [f"type1_{x}" for x in range(1, 4)],
        "type2": [f"type2_{x}" for x in range(1, 4)],
    }
    X_names = X_types["type1"] + X_types["type2"]

    # make df
    data = pd.DataFrame(X)
    data.columns = X_names
    data["target"] = y

    # create individual models
    model_1 = PipelineCreator(problem_type="regression", apply_to="type1")
    model_1.add("filter_columns", apply_to="*", keep="type1")
    model_1.add("svm", C=[1, 2])

    model_2 = PipelineCreator(problem_type="regression", apply_to="type2")
    model_2.add("filter_columns", apply_to="*", keep="type2")
    model_2.add("rf")

    # Create the stacking model
    model = PipelineCreator(problem_type="regression")
    model.add(
        "stacking",
        estimators=[[("model_1", model_1), ("model_2", model_2)]],
        apply_to="*",
    )

    # run
    _, final = run_cross_validation(
        X=X_names,
        X_types=X_types,
        y="target",
        data=data,
        model=model,
        seed=200,
        return_estimator="final",
    )

    # The final model should be a stacking model im which the first estimator
    # is a grid search
    assert isinstance(final.steps[1][1].model.estimators[0][1], GridSearchCV)


def test_inspection_error(df_iris):
    X = ["sepal_length", "sepal_width", "petal_length"]
    y = "species"
    with pytest.raises(ValueError, match="return_inspector=True requires"):
        run_cross_validation(
            X=X,
            y=y,
            data=df_iris,
            model="rf",
            return_estimator="final",
            return_inspector=True,
            problem_type="classification",
        )
    # default should be all now
    res = run_cross_validation(
        X=X,
        y=y,
        data=df_iris,
        model="rf",
        return_inspector=True,
        problem_type="classification",
    )
    assert len(res) == 3


def test_final_estimator_picklable(tmp_path, df_iris) -> None:
    X = ["sepal_length", "sepal_width", "petal_length"]
    y = "species"
    pickled_file = tmp_path / "final_estimator.joblib"
    _, final_estimator = run_cross_validation(
        X=X,
        y=y,
        data=df_iris,
        model="rf",
        problem_type="classification",
        return_estimator="final",
    )
    joblib.dump(final_estimator, pickled_file)
    # test if object can be loaded as well
    joblib.load(pickled_file)


def test_inspector_picklable(tmp_path, df_iris) -> None:
    X = ["sepal_length", "sepal_width", "petal_length"]
    y = "species"
    pickled_file = tmp_path / "inspector.joblib"
    _, _, inspector = run_cross_validation(
        X=X,
        y=y,
        data=df_iris,
        model="rf",
        problem_type="classification",
        return_estimator="all",
        return_inspector=True,
    )
    joblib.dump(inspector, pickled_file)
    # test if object can be loaded as well
    joblib.load(pickled_file)
