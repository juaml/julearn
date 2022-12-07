from sklearn.utils.metaestimators import available_if
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from . column_types import ColumnTypes


class JuBaseEstimator(BaseEstimator):
    def __init__(self, apply_to, needed_types=None):
        self.apply_to = apply_to
        self.needed_types = needed_types

    def get_needed_types(self):
        return self._ensure_needed_types()

    def get_apply_to(self):
        return self.apply_to

    def _ensure_apply_to(self):
        return self._ensure_column_types(self.apply_to)

    def _ensure_needed_types(self):
        needed_types = self._ensure_column_types(self.apply_to)
        if self.needed_types is not None:
            needed_types.add(self._ensure_column_types(self.needed_types))

        return needed_types

    def _ensure_column_types(self, attr):
        return ColumnTypes(attr)

    def filter_columns(self, X):
        self._filter = self.apply_to.to_type_selector()
        columns = self._filter(X)
        return X[columns]


class JuTransformer(JuBaseEstimator, TransformerMixin):

    def _add_backed_filtered(self, X, X_trans):
        filtered_columns = self._filter(X)
        non_filtered_columns = [
            col
            for col in list(X.columns)
            if col not in filtered_columns

        ]
        return pd.concat(
            (X.loc[:, non_filtered_columns], X_trans),
            axis=1
        )


def _wrapped_model_has(attr):

    def check(self):
        getattr(self.model_, attr)
        return True

    return check


class WrapModel(JuBaseEstimator):
    def __init__(self, model, apply_to=None, needed_types=None):
        self.model = model
        self.apply_to = apply_to
        self.needed_types = needed_types

    def fit(self, X, y=None, **fit_params):
        self.apply_to = ("continuous" if self.apply_to is None
                         else self.apply_to)
        self.apply_to = self._ensure_apply_to()
        self.needed_types = self._ensure_needed_types()

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

    @property
    def classes_(self):
        return self.model_.classes_

