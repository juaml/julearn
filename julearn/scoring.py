from sklearn.metrics._scorer import check_scoring


def get_extended_scorer(estimator, score_name):
    scorer = check_scoring(estimator, scoring=score_name)

    def extended_scorer(estimator, X, y):
        X_conf_trans = estimator.transform_confounds(X)
        y_true = estimator.transform_target(X_conf_trans, y)
        return scorer(estimator, X, y_true)
    return extended_scorer
