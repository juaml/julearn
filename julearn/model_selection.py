from sklearn.utils.validation import check_is_fitted


def wrap_search(searcher, *args, **kwargs):

    class wrap_searcher(searcher):

        def __init__(self):
            super().__init__(*args, **kwargs)

        def transform_confounds(self, X):
            check_is_fitted(self)
            return self.best_estimator_.transform_confounds(X)

        def transform_target(self, X, y):
            check_is_fitted(self)
            X_conf_trans = self.transform_confounds(X)
            return self.best_estimator_.transform_target(X_conf_trans, y)

    return wrap_searcher()
