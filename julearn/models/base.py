from sklearn.base import BaseEstimator
from sklearn.utils.metaestimators import available_if

from .. utils.column_types import ensure_apply_to, make_type_selector


class JuModel(BaseEstimator):
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
        filter = make_type_selector(self._apply_to)
        columns = filter(X)
        return X[columns]


def _wrapped_model_has(attr):

    def check(self):
        getattr(self.model_, attr)
        return True

    return check


class WrapModel(JuModel):
    def __init__(self, model, apply_to=None, needed_types=None):
        self.model = model
        self.apply_to = apply_to
        self.needed_types = needed_types

    def fit(self, X, y=None, **fit_params):
        self.apply_to = ("continuous" if self.apply_to is None
                         else self.apply_to)
        self.needed_types = (self.apply_to if self.needed_types is None
                             else self.needed_types
                             )
        self._apply_to = self._ensure_apply_to(self.apply_to)

        Xt = self.filter_columns(X)
        self.model_ = self.model
        self.model_.fit(Xt, y, **fit_params)
        return self

    def predict(self, X):
        Xt = self.filter_columns(X)
        return self.model_.predict(Xt)

    def score(self, X, y):
        Xt = self.filter_columns(X)
        return self.model_.score(Xt, y)

    @available_if(_wrapped_model_has("predict_proba"))
    def predict_proba(self, X):
        Xt = self.filter_columns(X)
        return self.model_.predict_proba(Xt)

    @available_if(_wrapped_model_has("predict_proba"))
    def decision_function(self, X):
        Xt = self.filter_columns(X)
        return self.model_.decision_function(Xt)
