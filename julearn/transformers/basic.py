from sklearn.base import TransformerMixin, BaseEstimator


class PassThroughTransformer(TransformerMixin, BaseEstimator):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X
