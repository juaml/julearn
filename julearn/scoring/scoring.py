# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL
from sklearn.metrics._scorer import check_scoring, _check_multimetric_scoring
from . available_scorers import get_scorer


def ju_check_scoring(estimator, scoring):
    if scoring is None:
        return scoring
    if isinstance(scoring, str) or callable(scoring):
        scoring = get_scorer(scoring)
        return check_scoring(estimator, scoring=scoring)
    if isinstance(scoring, list):
        scoring = {score: get_scorer(score) for score in scoring}

    return _check_multimetric_scoring(estimator, scoring)
