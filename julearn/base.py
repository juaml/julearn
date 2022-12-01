from . utils.column_types import ColumnTypes
from sklearn.base import BaseEstimator


class JuBaseEstimator(BaseEstimator):
    def __init__(self, apply_to, needed_types=None):
        self.apply_to = apply_to
        self.needed_types = needed_types

    def get_needed_types(self):
        return self.needed_types

    def get_apply_to(self):
        return self.apply_to

    def _ensure_apply_to(self):
        return self._ensure_column_types(self.apply_to)

    def _ensure_needed_types(self):
        if self.needed_types is None:
            needed_types = self._ensure_column_types(self.apply_to)
        else:
            needed_types = self._ensure_column_types(self.needed_types)
        return needed_types

    def _ensure_column_types(self, attr):
        return ColumnTypes(attr)

    def filter_columns(self, X):
        self._filter = self.apply_to.to_type_selector()
        columns = self._filter(X)
        return X[columns]
