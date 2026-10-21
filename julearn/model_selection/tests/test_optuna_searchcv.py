"""Provide tests for julearn's copy of OptunaSearchCV."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
# License: AGPL

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


optuna = pytest.importorskip("optuna")

from optuna.distributions import FloatDistribution  # noqa: E402
from sklearn.model_selection import KFold  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

from julearn import run_cross_validation  # noqa: E402
from julearn.external.optuna_searchcv import OptunaSearchCV  # noqa: E402
from julearn.pipeline import PipelineCreator  # noqa: E402


_X = ["sepal_length", "sepal_width", "petal_length", "petal_width"]


def test_OptunaSearchCV_predict(df_iris: pd.DataFrame) -> None:
    """Test that a fitted OptunaSearchCV can predict.

    Parameters
    ----------
    df_iris : pd.DataFrame
        The iris dataset.

    """
    search = OptunaSearchCV(
        SVC(),
        {"C": FloatDistribution(0.01, 100, log=True)},
        n_trials=3,
        random_state=42,
    )
    search.fit(df_iris[_X], df_iris["species"])
    best = search.best_estimator_
    np.testing.assert_array_equal(
        search.predict(df_iris[_X]), best.predict(df_iris[_X])
    )
    np.testing.assert_array_equal(
        search.decision_function(df_iris[_X]),
        best.decision_function(df_iris[_X]),
    )


def test_run_cross_validation_optuna(df_iris: pd.DataFrame) -> None:
    """Test cross-validation with hyperparameter tuning using optuna.

    Parameters
    ----------
    df_iris : pd.DataFrame
        The iris dataset.

    """
    creator = PipelineCreator(problem_type="classification")
    creator.add("zscore").add("svm", C=(0.01, 100, "log-uniform"))
    scores = run_cross_validation(
        X=_X,
        y="species",
        data=df_iris,
        model=creator,
        cv=3,
        scoring="accuracy",
        search_params={"kind": "optuna", "n_trials": 3},
    )
    assert not np.any(np.isnan(scores["test_score"]))  # type: ignore


def test_OptunaSearchCV_new_study_each_fit(df_iris: pd.DataFrame) -> None:
    """Test that each fit uses a new study if a study is given.

    Parameters
    ----------
    df_iris : pd.DataFrame
        The iris dataset.

    """
    study = optuna.create_study(study_name="julearn", direction="maximize")
    search = OptunaSearchCV(
        SVC(),
        {"C": FloatDistribution(0.01, 100, log=True)},
        n_trials=3,
        study=study,
    )
    for i_fit in range(2):
        search.fit(df_iris[_X], df_iris["species"])
        assert search.study_.study_name == f"julearn_fit{i_fit}"
        # The trials of previous fits are not reused
        assert len(search.trials_) == 3
    # The given study is not used for the trials
    assert len(study.trials) == 0


def test_run_cross_validation_optuna_study(
    df_iris: pd.DataFrame, tmp_path: Path
) -> None:
    """Test cross-validation with a given optuna study in a database.

    With a storage shared by the copies of the study, each fit must create
    its own study, so the folds do not share trials.

    Parameters
    ----------
    df_iris : pd.DataFrame
        The iris dataset.
    tmp_path : Path
        A temporary directory for the database.

    """
    storage = f"sqlite:///{tmp_path / 'study.db'}"
    study = optuna.create_study(
        study_name="julearn", direction="maximize", storage=storage
    )
    creator = PipelineCreator(problem_type="classification")
    creator.add("zscore").add("svm", C=(0.01, 100, "log-uniform"))
    scores = run_cross_validation(
        X=_X,
        y="species",
        data=df_iris,
        model=creator,
        cv=KFold(3, shuffle=True, random_state=42),
        scoring="accuracy",
        search_params={"kind": "optuna", "n_trials": 3, "study": study},
    )
    assert not np.any(np.isnan(scores["test_score"]))  # type: ignore
    n_trials = {
        summary.study_name: summary.n_trials
        for summary in optuna.study.get_all_study_summaries(storage)
    }
    assert n_trials == {
        "julearn": 0,
        "julearn_fit0": 3,
        "julearn_fit1": 3,
        "julearn_fit2": 3,
    }
