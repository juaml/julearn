"""Provide a scikit-learn Pipeline that can be scored with metadata routing."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
# License: AGPL

# TODO: Remove this module once julearn only supports scikit-learn versions
# with the fix for https://github.com/scikit-learn/scikit-learn/issues/35195,
# and use `sklearn.pipeline.Pipeline` again in `PipelineCreator.to_pipeline`.
#
# With metadata routing enabled (as julearn does), `Pipeline.score` passes
# `sample_weight=None` to the routing even if no sample weights are given.
# Routing is validated by parameter name, so scoring raises a `TypeError` if
# the final step does not route `sample_weight` for `score` (e.g. Bagging,
# Stacking and Voting estimators). Scikit-learn turns this error into NaN
# scores when cross-validating or tuning hyperparameters.

from functools import cache
from typing import Any

import numpy as np
from sklearn import config_context
from sklearn.pipeline import Pipeline
from sklearn.utils.metadata_routing import (
    _routing_enabled,
    get_routing_for_object,
    process_routing,
)
from sklearn.utils.validation import check_is_fitted

from ..utils import raise_error


class JuPipeline(Pipeline):
    """Scikit-learn Pipeline that can be scored with metadata routing.

    This is a :class:`sklearn.pipeline.Pipeline` that only passes
    ``sample_weight`` to the final step's ``score`` method if it is given, as
    when metadata routing is disabled. See
    https://github.com/scikit-learn/scikit-learn/issues/35195.

    """

    def score(
        self,
        X: Any,  # noqa: N803
        y: Any = None,
        sample_weight: Any = None,
        **params: Any,
    ) -> float:
        """Transform the data, and apply `score` with the final estimator.

        Parameters
        ----------
        X : iterable
            Data to predict on.
        y : iterable, optional
            Targets used for scoring (default None).
        sample_weight : array-like, optional
            If not None, this argument is passed as ``sample_weight`` keyword
            argument to the ``score`` method of the final estimator (default
            None).
        **params : dict of str -> object
            Parameters requested and accepted by steps.

        Returns
        -------
        float
            Result of calling `score` on the final estimator.

        """
        if not _routing_enabled():
            return super().score(X, y, sample_weight=sample_weight, **params)

        # Same as scikit-learn's Pipeline.score, but only routing
        # sample_weight if it is given.
        check_is_fitted(self)
        if sample_weight is not None:
            params["sample_weight"] = sample_weight
        routed_params = process_routing(self, "score", **params)
        Xt = X
        for _, name, transform in self._iter(with_final=False):
            Xt = transform.transform(Xt, **routed_params[name].transform)
        final_name, final_estimator = self.steps[-1]
        return final_estimator.score(Xt, y, **routed_params[final_name].score)


@cache
def _sklearn_pipeline_score_is_broken() -> bool:
    """Check if scikit-learn's Pipeline.score has the routing bug.

    Returns
    -------
    bool
        True if scoring a scikit-learn Pipeline fails with metadata routing
        enabled when no sample weights are given.

    """
    from sklearn.ensemble import VotingRegressor
    from sklearn.linear_model import LinearRegression

    X = np.arange(6.0).reshape(3, 2)
    y = np.arange(3.0)
    pipeline = Pipeline(
        [("model", VotingRegressor([("lr", LinearRegression())]))]
    )
    with config_context(enable_metadata_routing=True):
        pipeline.fit(X, y)
        try:
            pipeline.score(X, y)
        except TypeError:
            return True
    return False


def _iter_estimators(estimator: Any, _seen: set | None = None) -> Any:
    """Iterate over an estimator and all its nested estimators.

    Parameters
    ----------
    estimator : Any
        The estimator to iterate over.
    _seen : set, optional
        The ids of the objects already visited (default None).

    Yields
    ------
    Any
        The estimator and each of its nested estimators.

    """
    if _seen is None:
        _seen = set()
    if id(estimator) in _seen:
        return
    _seen.add(id(estimator))
    if isinstance(estimator, list | tuple):
        for item in estimator:
            yield from _iter_estimators(item, _seen)
        return
    if isinstance(estimator, type) or not hasattr(estimator, "get_params"):
        return
    yield estimator
    for value in estimator.get_params(deep=False).values():
        yield from _iter_estimators(value, _seen)


def check_pipelines_can_score(estimator: Any) -> None:
    """Check that the scikit-learn pipelines in an estimator can be scored.

    Pipelines created by julearn are :class:`.JuPipeline` objects, which can
    always be scored. A :class:`sklearn.pipeline.Pipeline` created by the user
    fails to score with metadata routing if its final step does not route
    ``sample_weight`` for ``score``, see
    https://github.com/scikit-learn/scikit-learn/issues/35195.

    Parameters
    ----------
    estimator : Any
        The estimator to check, including all its nested estimators.

    Raises
    ------
    ValueError
        If the estimator contains a scikit-learn Pipeline that cannot be
        scored.

    """
    if not _routing_enabled() or not _sklearn_pipeline_score_is_broken():
        return
    for t_estimator in _iter_estimators(estimator):
        if not isinstance(t_estimator, Pipeline) or isinstance(
            t_estimator, JuPipeline
        ):
            continue
        try:
            get_routing_for_object(t_estimator).validate_metadata(
                method="score", params={"sample_weight": None}
            )
        except TypeError:
            final_step = type(t_estimator.steps[-1][1]).__name__
            raise_error(
                "A scikit-learn Pipeline ending in a "
                f"{final_step} cannot be used in julearn with this version "
                "of scikit-learn: its score method fails with metadata "
                "routing (see https://github.com/scikit-learn/scikit-learn/"
                "issues/35195), giving NaN scores. Use a PipelineCreator to "
                "build this pipeline instead."
            )
