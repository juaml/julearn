from . utils.column_types import ensure_apply_to, make_type_selector
from sklearn.base import BaseEstimator


class JuBaseEstimator(BaseEstimator):
    def __init__(self, apply_to, needed_types=None):
        self.apply_to = apply_to
        self.needed_types = needed_types

    def get_needed_types(self):
        return ([] if self.needed_types is None else self.needed_types)

    def get_apply_to(self):
        return self.apply_to

    @staticmethod
    def _ensure_apply_to(apply_to):
        return ensure_apply_to(apply_to)

    def filter_columns(self, X):
        self._filter = make_type_selector(self.apply_to)
        columns = self._filter(X)
        return X[columns]
