"""Provide tests for pipeline and estimator inspectors."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL

from typing import Any, Dict, List, Optional, Type

import pandas as pd
import pytest
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from julearn.inspect import PipelineInspector, _EstimatorInspector
from julearn.pipeline import PipelineCreator
from julearn.transformers import JuColumnTransformer


class MockTestEst(BaseEstimator):
    """Class for estimator tests.

    Parameters
    ----------
    hype_0 : int
        First hyperparameter.
    hype_1 : int
        Second hyperparameter.

    """

    def __init__(self, hype_0: int = 0, hype_1: int = 1) -> None:
        self.hype_0 = hype_0
        self.hype_1 = hype_1

    def fit(
        self,
        X: pd.DataFrame,  # noqa: N803
        y: Optional[pd.Series] = None,
        **fit_params: Any,
    ) -> "MockTestEst":
        """Fit the estimator.

        Parameters
        ----------
        X : list of str
            The features to use.
        y : str, optional
            The target to use (default None).
        **fit_params : dict
            Parameters for fitting the estimator.

        Returns
        -------
        MockTestEst
            The fitted estimator.

        """
        self.param_0_ = 0
        self.param_1_ = 1
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        """Transform the estimator.

        Parameters
        ----------
        X : list of str
            The features to use.

        Returns
        -------
        list of str
            The transformed estimator.

        """
        return X


@pytest.mark.parametrize(
    "steps",
    [
        ["svm"],
        ["zscore", "svm"],
        ["pca", "svm"],
        ["zscore", "pca", "svm"],
    ],
)
def test_get_stepnames(steps: List[str], df_iris: pd.DataFrame) -> None:
    """Test step names fetch.

    Parameters
    ----------
    steps : list of str
        The parametrized step names.
    df_iris : pd.DataFrame
        The iris dataset.

    """
    pipe = (
        PipelineCreator(problem_type="classification")
        .from_list(steps, model_params={}, problem_type="classification")
        .to_pipeline()
    )
    pipe.fit(df_iris.iloc[:, :-1], df_iris.species)
    assert ["set_column_types", *steps] == PipelineInspector(
        pipe
    ).get_step_names()


@pytest.mark.parametrize(
    "steps,as_estimator,returns",
    [
        (["svm"], True, [SVC()]),
        (["zscore", "pca", "svm"], True, [StandardScaler(), PCA(), SVC()]),
        (["svm"], False, [_EstimatorInspector(SVC())]),
    ],
)
def test_steps(
    steps: List[str],
    as_estimator: bool,
    returns: List[Type],
    df_iris: "pd.DataFrame",
) -> None:
    """Test steps.

    Parameters
    ----------
    steps : list of str
        The parametrized step names.
    as_estimator : bool
        The parametrized flag to indicate whether to use as estimator.
    returns : list
        The parametrized list of object instances to return.
    df_iris : pd.DataFrame
        The iris dataset.

    """
    pipe = (
        PipelineCreator(problem_type="classification")
        .from_list(steps, model_params={}, problem_type="classification")
        .to_pipeline()
    )
    pipe.fit(df_iris.iloc[:, :-1], df_iris.species)
    inspector = PipelineInspector(pipe)
    for i, _ in enumerate(steps):
        step_est = inspector.get_step(name=steps[i], as_estimator=as_estimator)
        assert isinstance(step_est, returns[i].__class__)


@pytest.mark.parametrize(
    "est,fitted_params",
    [
        [
            MockTestEst(),
            {"hype_0": 0, "hype_1": 1, "param_0_": 0, "param_1_": 1},
        ],
        [
            JuColumnTransformer(
                "test",
                MockTestEst(),  # type: ignore
                "continuous",
            ),
            {
                "hype_0": 0,
                "hype_1": 1,
                "param_0_": 0,
                "param_1_": 1,
                "needed_types": None,
                "row_select_col_type": None,
                "row_select_vals": None,
            },
        ],
    ],
)
def test_inspect_estimator(
    est: Type, fitted_params: Dict[str, int], df_iris: "pd.DataFrame"
) -> None:
    """Test estimator inspector.

    Parameters
    ----------
    est : Estimator-like
        Estimator-like object.
    fitted_params : dict of str and int
        The fitted parameters for ``est``.
    df_iris : pd.DataFrame
        The iris dataset.

    """
    est.fit(df_iris.iloc[:, :-1], df_iris.species)
    inspector = _EstimatorInspector(est)
    assert est.get_params() == inspector.get_params()
    inspect_params = inspector.get_fitted_params()
    inspect_params.pop("column_transformer_", None)
    inspect_params.pop("apply_to", None)
    inspect_params.pop("transformer", None)
    inspect_params.pop("name", None)
    assert fitted_params == inspect_params


def test_inspect_pipeline(df_iris: "pd.DataFrame") -> None:
    """Test pipeline inspector.

    Parameters
    ----------
    df_iris : pd.DataFrame
        The iris dataset.

    """
    expected_fitted_params = {
        "jucolumntransformer__hype_0": 0,
        "jucolumntransformer__hype_1": 1,
        "jucolumntransformer__param_0_": 0,
        "jucolumntransformer__param_1_": 1,
        "jucolumntransformer__needed_types": None,
        "jucolumntransformer__row_select_col_type": None,
        "jucolumntransformer__row_select_vals": None,
        "jucolumntransformer__name": "test",
    }

    pipe = (
        PipelineCreator(problem_type="classification")
        .add(
            JuColumnTransformer(
                "test",
                MockTestEst(),  # type: ignore
                "continuous",
            )
        )
        .add(SVC())  # type: ignore TODO: fix typing hints
        .to_pipeline()
    )
    pipe.fit(df_iris.iloc[:, :-1], df_iris.species)
    inspector = PipelineInspector(pipe)
    inspect_params = inspector.get_fitted_params()
    inspect_params.pop("jucolumntransformer__column_transformer_", None)
    inspect_params.pop("jucolumntransformer__transformer", None)
    inspect_params.pop("jucolumntransformer__apply_to", None)
    inspect_params = {
        key: val
        for key, val in inspect_params.items()
        if (not key.startswith("svc"))
        and (not key.startswith("set_column_types"))
    }

    assert expected_fitted_params == inspect_params


def test_get_estimator(df_iris: "pd.DataFrame") -> None:
    """Test estimator fetch from inspector.

    Parameters
    ----------
    df_iris : pd.DataFrame
        The iris dataset.

    """
    pipe = (
        PipelineCreator(problem_type="classification")
        .add(
            JuColumnTransformer(
                "test",
                MockTestEst(),  # type: ignore
                "continuous",
            )
        )
        .add(SVC())  # type: ignore TODO: fix typing hints
        .to_pipeline()
    )
    pipe.fit(df_iris.iloc[:, :-1], df_iris.species)
    inspector = PipelineInspector(pipe)
    svc = inspector.get_step("svc").estimator
    assert isinstance(svc, SVC)
    assert pipe.get_params() == inspector.get_params()
