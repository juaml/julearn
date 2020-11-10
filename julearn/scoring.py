# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL
from sklearn.metrics._scorer import check_scoring


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
    scorer = check_scoring(estimator, scoring=score_name)

    def extended_scorer(estimator, X, y):
        y_true = estimator.transform_target(X, y)
        return scorer(estimator, X, y_true)
    return extended_scorer
