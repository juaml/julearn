"""Provide registry of julearn's scorers."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL

import typing
import warnings
from copy import deepcopy
from typing import Callable, Dict, List, Optional, Union

from sklearn.metrics import _scorer, get_scorer_names, make_scorer
from sklearn.metrics._scorer import _check_multimetric_scoring
from sklearn.metrics._scorer import check_scoring as sklearn_check_scoring

from ..transformers.target.ju_transformed_target_model import (
    TransformedTargetWarning,
)
from ..utils import logger, raise_error, warn_with_log
from ..utils.typing import EstimatorLike, ScorerLike
from .metrics import r2_corr, r_corr


_extra_available_scorers = {
    "r2_corr": make_scorer(r2_corr),
    "r_corr": make_scorer(r_corr),
}

_extra_available_scorers_reset = deepcopy(_extra_available_scorers)


def get_scorer(name: str) -> ScorerLike:
    """Get available scorer by name.

    Parameters
    ----------
    name : str
        name of an available scorer

    Returns
    -------
    scorer : ScorerLike
        Callable object that returns a scalar score; greater is better.
        Will be called using `(estimator, X, y)`.
    """
    scorer = _extra_available_scorers.get(name)
    if scorer is None:
        try:
            scorer = _scorer.get_scorer(name)
        except ValueError:
            raise_error(
                f"{name} is not a valid scorer "
                "please use julearn.scorers.list_scorers to get a list"
                "of possible scorers"
            )
    return scorer


def list_scorers() -> List[str]:
    """List all available scorers.

    Returns
    -------
    list of str
        a list containing all available scorers.
    """
    scorers = list(get_scorer_names())
    scorers.extend(list(_extra_available_scorers.keys()))
    return scorers


def register_scorer(
    scorer_name: str, scorer: ScorerLike, overwrite: Optional[bool] = None
) -> None:
    """Register a scorer, so that it can be accessed by name.

    Parameters
    ----------
    scorer_name : str
        name of the scorer you want to register
    scorer : ScorerLike
        Callable object that returns a scalar score; greater is better.
        Will be called using `(estimator, X, y)`.
    overwrite : bool, optional
        decides whether overwrite should be allowed. Options are:

        * None : overwrite is possible, but warns the user
        * True : overwrite is possible without any warning
        * False : overwrite is not possible, error is raised instead

        (default is None)

    Raises
    ------
    ValueError
        if overwrite is set to False and the scorer already exists.

    Warns
    -----
    UserWarning
        if overwrite is set to None and the scorer already exists.
    """
    if scorer_name in list_scorers():
        if overwrite is None:
            warn_with_log(
                f"scorer named {scorer_name} already exists. "
                f"Therefore, {scorer_name} will be overwritten. "
                "To remove this warning set overwrite=True "
            )
            logger.info(f"registering scorer named {scorer_name}")
        elif overwrite is False:
            raise_error(
                f"scorer named {scorer_name} already exists and "
                "overwrite is set to False, therefore you cannot overwrite "
                "existing scorers. Set overwrite=True in case you want to "
                "overwrite existing scorers"
            )
    logger.info(f"registering scorer named {scorer_name}")
    _extra_available_scorers[scorer_name] = scorer


def reset_scorer_register():
    """Reset the scorer register to the default state."""
    global _extra_available_scorers
    _extra_available_scorers = deepcopy(_extra_available_scorers_reset)


def check_scoring(
    estimator: EstimatorLike,
    scoring: Union[ScorerLike, str, Callable, List[str], None],
    wrap_score: bool,
) -> Union[None, ScorerLike, Callable, Dict[str, ScorerLike]]:
    """Check the scoring.

    Parameters
    ----------
    estimator : EstimatorLike
        estimator to check the scoring for
    scoring : Union[ScorerLike, str, Callable]
        scoring to check
    wrap_score : bool
        Does the score needs to be wrapped
        to handle non_inverse transformable target pipelines.
    """
    if scoring is None:
        return scoring
    if isinstance(scoring, str):
        scoring = _extend_scorer(get_scorer(scoring), wrap_score)
    if callable(scoring):
        return _extend_scorer(
            sklearn_check_scoring(estimator, scoring=scoring), wrap_score
        )
    if isinstance(scoring, list):
        scorer_names = typing.cast(List[str], scoring)
        scoring_dict = {
            score: _extend_scorer(get_scorer(score), wrap_score)
            for score in scorer_names
        }
        return _check_multimetric_scoring(  # type: ignore
            estimator, scoring_dict
        )


def _extend_scorer(scorer, extend):
    if extend:
        return _ExtendedScorer(scorer)
    return scorer


class _ExtendedScorer:
    def __init__(self, scorer):
        self.scorer = scorer

    def __call__(self, estimator, X, y):  # noqa: N803
        if hasattr(estimator, "best_estimator_"):
            estimator = estimator.best_estimator_

        X_trans = X
        for _, transform in estimator.steps[:-1]:
            X_trans = transform.transform(X_trans)
        y_true = estimator.steps[-1][-1].transform_target(  # last est
            X_trans, y
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                action="ignore", category=TransformedTargetWarning
            )
            scores = self.scorer(estimator, X, y_true)
        return scores
