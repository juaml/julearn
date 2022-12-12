from ...base import ColumnTypes


class JuTargetTransformer:
    def _ensure_column_types(self, attr):
        return ColumnTypes(attr)

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X, y)

    def fit(self, X, y):
        raise NotImplementedError("fit method not implemented")

    def transform(self, X, y):
        raise NotImplementedError("fitransform method not implemented")
