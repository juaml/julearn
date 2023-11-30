"""Provide conftest for pytest."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL
import typing
from copy import copy
from typing import Callable, Dict, List, Optional, Union

import pandas as pd
from pytest import FixtureRequest, fixture
from seaborn import load_dataset


@fixture(scope="function")
def df_typed_iris() -> pd.DataFrame:
    """Return a typed iris dataset.

    Returns
    -------
    df : pd.DataFrame
        The iris dataset with types.
    """
    df = load_dataset("iris")
    df = typing.cast(pd.DataFrame, df)

    rename = {
        "sepal_length": "sepal_length__:type:__continuous",
        "sepal_width": "sepal_width__:type:__continuous",
        "petal_length": "petal_length__:type:__continuous",
        "petal_width": "petal_width__:type:__continuous",
    }

    return df.rename(columns=rename)


@fixture(scope="function")
def df_iris() -> pd.DataFrame:
    """Return the iris dataset.

    Returns
    -------
    df : pd.DataFrame
        The iris dataset with types.
    """
    df = load_dataset("iris")
    df = typing.cast(pd.DataFrame, df)

    return df.copy()


@fixture(scope="function")
def df_binary() -> pd.DataFrame:
    """Return the iris dataset as a binary classification problem.

    Returns
    -------
    df : pd.DataFrame
        The iris dataset with types.
    """
    df_iris = load_dataset("iris")
    df_iris = typing.cast(pd.DataFrame, df_iris)
    df_binary = df_iris[df_iris["species"].isin(["setosa", "virginica"])]

    return df_binary


@fixture(scope="function")
def X_iris() -> pd.DataFrame:
    """Return the iris dataset features.

    Features are "sepal_length", "sepal_width", "petal_length", "petal_width".

    Returns
    -------
    df : pd.DataFrame
        The iris dataset features.
    """
    df = load_dataset("iris")
    df = typing.cast(pd.DataFrame, df)
    features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    return df.loc[:, features]


@fixture(scope="function")
def y_iris() -> pd.Series:
    """Return the iris dataset target as integers.

    Target is "species".

    Returns
    -------
    df : pd.Series
        The iris dataset target.
    """
    df = load_dataset("iris")
    df = typing.cast(pd.DataFrame, df)

    return df.loc[:, "species"].astype("category").cat.codes


@fixture(
    params=[
        None,
        {},
        {"duck": ["petal_length"]},
        {"duck": ["petal_length"], "confound": ["petal_width"]},
    ],
    scope="function",
)
def X_types_iris(request: FixtureRequest) -> Optional[Dict]:
    """Return different types for the iris dataset features.

    Parameters
    ----------
    request : pytest.FixtureRequest
        The request object.

    Returns
    -------
    dict or None
        A dictionary with the types for the features.
    """
    return request.param


@fixture(params=["rf", "svm", "gauss", "ridge"], scope="function")
def models_all_problem_types(request: FixtureRequest) -> str:
    """Return different models that work with classification and regression.

    Parameters
    ----------
    request : pytest.FixtureRequest
        The request object.

    Returns
    -------
    str
        The name of the model.
    """
    return request.param


@fixture(params=["regression", "classification"], scope="function")
def all_problem_types(request: FixtureRequest) -> str:
    """Return different problem types.

    Parameters
    ----------
    request : pytest.FixtureRequest
        The request object.

    Returns
    -------
    str
        The problem type (one of {"regression", "classification"}).
    """

    return request.param


@fixture(
    params=[
        None,
        {},
        {"kind": "grid"},
        {"kind": "random", "n_iter": 2},
        {"kind": "random", "n_iter": 2, "cv": 3},
    ],
    scope="function",
)
def search_params(request: FixtureRequest) -> Optional[Dict]:
    """Return different possibiblites for the search_params argument.

    Parameters
    ----------
    request : pytest.FixtureRequest
        The request object.

    Returns
    -------
    dict or None
        A dictionary with the search_params argument.
    """
    return request.param


_tuning_params = {
    "zscore": {"with_mean": [True, False]},
    "pca": {"n_components": [0.2, 0.7]},
    "select_univariate": {"mode": ["k_best", "percentile"]},
    "rf": {"n_estimators": [2, 5]},
    "svm": {"C": [1, 2]},
    "ridge": {"alpha": [1, 2]},
}


@fixture(scope="function")
def get_tuning_params() -> Callable:
    """Return a function that returns the parameters to tune for a given step.

    Returns
    -------
    get : callable
        A function that returns the parameters to tune for a given step.
    """

    def get(step: str) -> Dict:
        """Return the parameters to tune for a given step.

        Parameters
        ----------
        step : str
            The name of the step.

        Returns
        -------
        dict
            The parameters to tune for the given step.
        """
        return copy(_tuning_params.get(step, {}))

    return get


@fixture(
    params=[
        "zscore",
        ["zscore"],
        ["pca"],
        ["select_univariate"],
        ["zscore", "pca"],
        ["select_univariate", "zscore", "pca"],
    ],
    scope="function",
)
def preprocessing(request: FixtureRequest) -> Union[str, List[str]]:
    """Return different preprocessing steps.

    Parameters
    ----------
    request : pytest.FixtureRequest
        The request object.

    Returns
    -------
    str or list
        The preprocessing step(s).
    """
    return request.param
