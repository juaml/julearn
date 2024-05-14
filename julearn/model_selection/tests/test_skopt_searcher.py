"""Provides tests for the bayes searcher."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
# License: AGPL

from typing import Any, Dict, Tuple

import pytest

from julearn.model_selection._skopt_searcher import (
    _prepare_skopt_hyperparameters_distributions,
)


sksp = pytest.importorskip("skopt.space")


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
            ("int", "int", "int"),
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
                "n_components": sksp.Real(0.2, 0.7, prior="uniform"),
                "n_neighbors": sksp.Real(1.0, 10.0, prior="log-uniform"),
            },
            ("float", "float"),
            ("uniform", "log-uniform"),
        ),
        (
            {
                "n_components": sksp.Integer(1, 20, prior="uniform"),
                "n_neighbors": sksp.Integer(1, 10, prior="log-uniform"),
            },
            ("int", "int"),
            ("uniform", "log-uniform"),
        ),
        (
            {
                "options": sksp.Categorical([True, False]),
                "more_options": sksp.Categorical(
                    ("a", "b", "c", "d"),
                ),
            },
            (None, None),
            ("categorical", "categorical"),
        ),
    ],
)
def test__prepare_skopt_hyperparameters_distributions(
    params_to_tune: Dict[str, Dict[str, Any]],
    expected_types: Tuple,
    expected_dist: Tuple,
) -> None:
    """Test the _prepare_skopt_hyperparameters_distributions function.

    Parameters
    ----------
    params_to_tune : dict
        The parameters to tune.
    expected_types : tuple
        The expected types of each parameter.
    expected_dist : tuple
        The expected distributions of each parameter.

    """
    new_params = _prepare_skopt_hyperparameters_distributions(params_to_tune)
    for i, (k, v) in enumerate(new_params.items()):
        if expected_types[i] == "int":
            assert isinstance(v, sksp.Integer)
            assert v.prior == expected_dist[i]
            if isinstance(params_to_tune[k], tuple):
                assert v.bounds[0] == params_to_tune[k][0]  # type: ignore
                assert v.bounds[1] == params_to_tune[k][1]  # type: ignore
            else:
                assert isinstance(params_to_tune[k], sksp.Integer)
                assert v.bounds[0] == params_to_tune[k].bounds[0]  # type: ignore
                assert v.bounds[1] == params_to_tune[k].bounds[1]  # type: ignore
                assert params_to_tune[k].prior == v.prior  # type: ignore
        elif expected_types[i] == "float":
            assert isinstance(v, sksp.Real)
            assert v.prior == expected_dist[i]
            if isinstance(params_to_tune[k], tuple):
                assert v.bounds[0] == params_to_tune[k][0]  # type: ignore
                assert v.bounds[1] == params_to_tune[k][1]  # type: ignore
            else:
                assert isinstance(params_to_tune[k], sksp.Real)
                assert v.bounds[0] == params_to_tune[k].bounds[0]  # type: ignore
                assert v.bounds[1] == params_to_tune[k].bounds[1]  # type: ignore
                assert params_to_tune[k].prior == v.prior  # type: ignore
        elif expected_dist[i] == "categorical":
            assert isinstance(v, sksp.Categorical)
            if isinstance(params_to_tune[k], tuple):
                assert all(
                    x in v.categories
                    for x in params_to_tune[k][:-1]  # type: ignore
                )
                assert all(
                    x in params_to_tune[k][:-1]  # type: ignore
                    for x in v.categories
                )
            else:
                assert isinstance(params_to_tune[k], sksp.Categorical)
                assert all(
                    x in v.categories
                    for x in params_to_tune[k].categories  # type: ignore
                )
                assert all(
                    x in params_to_tune[k].categories  # type: ignore
                    for x in v.categories
                )
        else:
            pytest.fail("Invalid distribution type")
