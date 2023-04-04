"""Provide registry of julearn's scorers."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL

import typing
from copy import deepcopy
from typing import Callable, Dict, List, Optional, Union

from sklearn.metrics import SCORERS, _scorer, make_scorer
from sklearn.metrics._scorer import _check_multimetric_scoring
from sklearn.metrics._scorer import check_scoring as sklearn_check_scoring

from ..utils import logger, raise_error, warn_with_log
from ..utils.typing import EstimatorLike, ScorerLike
from .metrics import r2_corr


_extra_available_scorers = {"r2_corr": make_scorer(r2_corr)}

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
    scorers = list(SCORERS.keys())
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
) -> Union[None, ScorerLike, Callable, Dict[str, ScorerLike]]:
    """Check the scoring.

    Parameters
    ----------
    estimator : EstimatorLike
        estimator to check the scoring for
    scoring : Union[ScorerLike, str, Callable]
        scoring to check
    """
    if scoring is None:
        return scoring
    if isinstance(scoring, str):
        scoring = get_scorer(scoring)
    if callable(scoring):
        return sklearn_check_scoring(estimator, scoring=scoring)
    if isinstance(scoring, list):
        scorer_names = typing.cast(List[str], scoring)
        scoring_dict = {score: get_scorer(score) for score in scorer_names}
        return _check_multimetric_scoring(  # type: ignore
            estimator, scoring_dict
        )
