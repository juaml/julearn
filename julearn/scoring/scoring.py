# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL
from sklearn.metrics._scorer import check_scoring
from . available_scorers import get_scorer


def _check_scoring(estimator, scoring):
    if isinstance(scoring, str):
        scoring = get_scorer(scoring)
    elif isinstance(scoring, list):
        scoring = {score: get_scorer(score) for score in scoring}
    return check_scoring(estimator, scoring=scoring)
