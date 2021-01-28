# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL
from sklearn.metrics._scorer import check_scoring
from . available_scorers import get_scorer


def get_extended_scorer(estimator, score_name):
    """A function that can use an estimator and score_name to
    create a scorer compatible with julearn.pipeline.ExtendedDataFramePipeline.

    Parameters
    ----------
    estimator : julearn.pipeline.ExtendedDataFramePipeline
        An estimator with a .transform_confounds and .transform_target
        method needed for scoring against a new ground truth
    score_name : str
        The name of the score you want to use for scoring.
        All scores available in sklearn are compatible

    Returns
    -------
    extended_scorer : sklearn scorer
        A function with arguments: estimator, X, y .
        That returns a single score.
    """

    scorer = _check_scoring(estimator, scoring=score_name)
    return _ExtendedScorer(scorer)


def _check_scoring(estimator, scoring):
    if isinstance(scoring, str):
        scoring = get_scorer(scoring)
    elif isinstance(scoring, list):
        scoring = {score: get_scorer(score) for score in scoring}
    return check_scoring(estimator, scoring=scoring)


class _ExtendedScorer():
    def __init__(self, scorer):
        self.scorer = scorer

    def __call__(self, estimator, X, y):
        if hasattr(estimator, 'transform_target'):
            y_true = estimator.transform_target(X, y)
        else:
            y_true = estimator.best_estimator_.transform_target(X, y)

        return self.scorer(estimator, X, y_true)
