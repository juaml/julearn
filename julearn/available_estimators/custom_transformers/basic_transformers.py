from sklearn.base import TransformerMixin, BaseEstimator


class PassThroughTransformer(TransformerMixin, BaseEstimator):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X


class TargetPassThroughTransformer(PassThroughTransformer):
    def __init__(self):
        super().__init__()

    def transform(self, X=None, y=None):
        return y

    def fit_transform(self, X=None, y=None):
        self.fit(X, y)
        return self.transform(X, y)
