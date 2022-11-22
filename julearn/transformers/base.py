from sklearn.base import BaseEstimator, TransformerMixin
from ..utils.column_types import ensure_apply_to


class JuTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, apply_to, needed_types=None):
        self.apply_to = apply_to
        self.needed_types = needed_types

    def get_needed_types(self):
        return self.needed_types

    def get_apply_to(self):
        return self.apply_to

    @staticmethod
    def _ensure_apply_to(apply_to):
        return ensure_apply_to(apply_to)
