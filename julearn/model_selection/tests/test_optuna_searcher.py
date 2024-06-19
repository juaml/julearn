"""Provides tests for the optuna searcher."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
# License: AGPL

from typing import Any, Dict, Tuple

import pytest

from julearn.model_selection._optuna_searcher import (
    _prepare_optuna_hyperparameters_distributions,
    is_optuna_valid_distribution,
)


optd = pytest.importorskip("optuna.distributions")


@pytest.mark.parametrize(
    "params_to_tune,expected_types, expected_dist",
    [
        (
            {
                "n_components": (0.2, 0.7, "uniform"),
                "n_neighbors": (1.0, 10.0, "log-uniform"),
            },
            ("float", "float"),
            ("uniform", "log-uniform"),
        ),
        (
            {
                "n_components": (1, 20, "uniform"),
                "n_neighbors": (1, 10, "log-uniform"),
            },
            ("int", "int"),
            ("uniform", "log-uniform"),
        ),
        (
            {
                "options": (True, False, "categorical"),
                "more_options": ("a", "b", "c", "d", "categorical"),
            },
            (None, None),
            ("categorical", "categorical"),
        ),
        (
            {
                "n_components": optd.FloatDistribution(0.2, 0.7, log=False),
                "n_neighbors": optd.FloatDistribution(1.0, 10.0, log=True),
            },
            ("float", "float"),
            ("uniform", "log-uniform"),
        ),
        (
            {
                "n_components": optd.IntDistribution(1, 20, log=False),
                "n_neighbors": optd.IntDistribution(1, 10, log=True),
            },
            ("int", "int"),
            ("uniform", "log-uniform"),
        ),
        (
            {
                "options": optd.CategoricalDistribution([True, False]),
                "more_options": optd.CategoricalDistribution(
                    ("a", "b", "c", "d"),
                ),
            },
            (None, None),
            ("categorical", "categorical"),
        ),
    ],
)
def test__prepare_optuna_hyperparameters_distributions(
    params_to_tune: Dict[str, Dict[str, Any]],
    expected_types: Tuple,
    expected_dist: Tuple,
) -> None:
    """Test the _prepare_optuna_hyperparameters_distributions function.

    Parameters
    ----------
    params_to_tune : dict
        The parameters to tune.
    expected_types : tuple
        The expected types of each parameter.
    expected_dist : tuple
        The expected distributions of each parameter.

    """
    new_params = _prepare_optuna_hyperparameters_distributions(params_to_tune)
    for i, (k, v) in enumerate(new_params.items()):
        if expected_dist[i] == "uniform":
            if expected_types[i] == "int":
                assert isinstance(v, optd.IntDistribution)
                assert not v.log
                if isinstance(params_to_tune[k], tuple):
                    assert v.low == params_to_tune[k][0]  # type: ignore
                    assert v.high == params_to_tune[k][1]  # type: ignore
                else:
                    assert isinstance(params_to_tune[k], optd.IntDistribution)
                    assert v.low == params_to_tune[k].low  # type: ignore
                    assert v.high == params_to_tune[k].high  # type: ignore
                    assert not params_to_tune[k].log  # type: ignore
            else:
                assert isinstance(v, optd.FloatDistribution)
                assert not v.log
                if isinstance(params_to_tune[k], tuple):
                    assert v.low == params_to_tune[k][0]  # type: ignore
                    assert v.high == params_to_tune[k][1]  # type: ignore
                else:
                    assert isinstance(
                        params_to_tune[k], optd.FloatDistribution
                    )
                    assert v.low == params_to_tune[k].low  # type: ignore
                    assert v.high == params_to_tune[k].high  # type: ignore
                    assert not params_to_tune[k].log  # type: ignore
        elif expected_dist[i] == "log-uniform":
            if expected_types[i] == "int":
                assert isinstance(v, optd.IntDistribution)
                assert v.log
                if isinstance(params_to_tune[k], tuple):
                    assert v.low == params_to_tune[k][0]  # type: ignore
                    assert v.high == params_to_tune[k][1]  # type: ignore
                else:
                    assert isinstance(params_to_tune[k], optd.IntDistribution)
                    assert v.low == params_to_tune[k].low  # type: ignore
                    assert v.high == params_to_tune[k].high  # type: ignore
                    assert params_to_tune[k].log  # type: ignore
            else:
                assert isinstance(v, optd.FloatDistribution)
                assert v.log
                if isinstance(params_to_tune[k], tuple):
                    assert v.low == params_to_tune[k][0]  # type: ignore
                    assert v.high == params_to_tune[k][1]  # type: ignore
                else:
                    assert isinstance(
                        params_to_tune[k], optd.FloatDistribution
                    )
                    assert v.low == params_to_tune[k].low  # type: ignore
                    assert v.high == params_to_tune[k].high  # type: ignore
                    assert params_to_tune[k].log  # type: ignore
        elif expected_dist[i] == "categorical":
            assert isinstance(v, optd.CategoricalDistribution)
            if isinstance(params_to_tune[k], tuple):
                assert all(
                    x in v.choices
                    for x in params_to_tune[k][:-1]  # type: ignore
                )
                assert all(
                    x in params_to_tune[k][:-1]  # type: ignore
                    for x in v.choices
                )
            else:
                assert isinstance(
                    params_to_tune[k], optd.CategoricalDistribution
                )
                assert all(
                    x in v.choices
                    for x in params_to_tune[k].choices  # type: ignore
                )
                assert all(
                    x in params_to_tune[k].choices  # type: ignore
                    for x in v.choices
                )
        else:
            pytest.fail("Invalid distribution type")


@pytest.mark.parametrize(
    "obj,expected",
    [
        (optd.IntDistribution(1, 20, log=False), True),
        (optd.FloatDistribution(0.2, 0.7, log=False), True),
        (optd.CategoricalDistribution([1, 2, 3]), True),
        (optd.CategoricalDistribution(["a", "b", "c"]), True),
        (optd.CategoricalDistribution(["a", "b", "c", "d"]), True),
        ("uniform", False),
        ("log-uniform", False),
        ("categorical", False),
        (1, False),
        (1.0, False),
        ([1, 2, 3], False),
        (["a", "b", "c"], False),
    ],
)
def test_optuna_valid_distributions(obj: Any, expected: bool) -> None:
    """Test the optuna_valid_distributions function.

    Parameters
    ----------
    obj : Any
        The object to check.
    expected : bool
        The expected result.

    """
    out = is_optuna_valid_distribution(obj)
    assert out == expected
