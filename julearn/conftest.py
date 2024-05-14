"""Provide conftest for pytest."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL
import typing
from copy import copy
from typing import Callable, Dict, List, Optional, Union

import pandas as pd
import pytest
from pytest import FixtureRequest, fixture, mark
from seaborn import load_dataset


_filter_keys = {
    "nodeps": "Test that runs without conditional dependencies only",
}


def pytest_configure(config: pytest.Config) -> None:
    """Add a new marker to pytest.

    Parameters
    ----------
    config : pytest.Config
        The pytest configuration object.

    """
    # register your new marker to avoid warnings
    for k, v in _filter_keys.items():
        config.addinivalue_line("markers", f"{k}: {v}")


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add a new filter option to pytest.

    Parameters
    ----------
    parser : pytest.Parser
        The pytest parser object.

    """
    # add your new filter option (you can name it whatever you want)
    parser.addoption(
        "--filter",
        action="store",
        help="Select tests based on markers.",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: List[pytest.Item]
) -> None:
    """Filter tests based on the key marker.

    Parameters
    ----------
    config : pytest.Config
        The pytest configuration object.
    items : list
        The list of items.

    """
    filter = config.getoption("--filter", None)  # type: ignore
    if filter is None:
        for k in _filter_keys.keys():
            skip_keys = mark.skip(
                reason=f"Filter not specified for this test: {k}"
            )
            for item in items:
                if k in item.keywords:
                    item.add_marker(skip_keys)  # skip the test
    else:
        new_items = []
        for item in items:
            if filter in item.keywords:
                new_items.append(item)
        items[:] = new_items


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
def model(request: FixtureRequest) -> str:
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
def problem_type(request: FixtureRequest) -> str:
    """Return different problem types.

    Parameters
    ----------
    request : pytest.FixtureRequest
        The request object.

    Returns
    -------
    str
        The problem type.

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


@fixture(
    params=[
        {"kind": "bayes", "n_iter": 2, "cv": 3},
        {"kind": "bayes", "n_iter": 2},
    ],
    scope="function",
)
def bayes_search_params(request: FixtureRequest) -> Optional[Dict]:
    """Return different search_params argument for BayesSearchCV.

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


@fixture(
    params=[
        {"kind": "optuna", "n_trials": 10, "cv": 3},
        {"kind": "optuna", "timeout": 20},
    ],
    scope="function",
)
def optuna_search_params(request: FixtureRequest) -> Optional[Dict]:
    """Return different search_params argument for OptunaSearchCV.

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


_tuning_distributions = {
    "zscore": {"with_mean": [True, False]},
    "pca": {"n_components": (0.2, 0.7, "uniform")},
    "select_univariate": {"mode": ["k_best", "percentile"]},
    "rf": {"n_estimators": [2, 5]},
    "svm": {"C": (1, 10, "log-uniform")},
    "ridge": {"alpha": (1, 3, "uniform")},
}


@fixture(scope="function")
def get_tuning_distributions() -> Callable:
    """Return a function that returns the distributions to tune.

    Returns
    -------
    get : callable
        A function that returns the distributions to tune for a given step.

    """

    def get(step: str) -> Dict:
        """Return the distributions to tune for a given step.

        Parameters
        ----------
        step : str
            The name of the step.

        Returns
        -------
        dict
            The distributions to tune for the given step.

        """
        return copy(_tuning_distributions.get(step, {}))

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
def preprocess(request: FixtureRequest) -> Union[str, List[str]]:
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
