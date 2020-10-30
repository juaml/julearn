from sklearn.metrics._scorer import check_scoring


def get_extended_scorer(estimator, score_name):
    """A function that can use an estimator and score_name to
    create a scorer compatible with julearn.pipeline.ExtendedDataFramePipeline.

    Parameters
    ----------
    estimator : julearn.pipeline.ExtendedDataFramePipeline
        A estimator with a .transform_confounds and .transform_target
        method needed for scoring against a new ground truth
    score_name : str
        The name of the score you want to use for scoring.
        All scores available in sklearn are compatible

    Returns
    -------
    sklearn scorer
        A function with arguments: estimator, X, y .
        That returns a single score.
    """
    scorer = check_scoring(estimator, scoring=score_name)

    def extended_scorer(estimator, X, y):
        X_conf_trans = estimator.transform_confounds(X)
        y_true = estimator.transform_target(X_conf_trans, y)
        return scorer(estimator, X, y_true)
    return extended_scorer
