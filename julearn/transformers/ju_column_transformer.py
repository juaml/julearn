from sklearn.compose import ColumnTransformer
from sklearn.utils.validation import check_is_fitted

from .. base import JuTransformer


class JuColumnTransformer(JuTransformer):

    def __init__(self, name, transformer, apply_to, needed_types=None):
        self.name = name
        self.transformer = transformer
        self.apply_to = apply_to
        self.needed_types = needed_types

    def fit(self, X, y=None, **fit_params):
        self._ensure_apply_to()
        self._ensure_needed_types()

        self.column_transformer_ = ColumnTransformer(
            [(self.name, self.transformer, self.apply_to.to_type_selector())],
            verbose_feature_names_out=False,
            remainder="passthrough",
        )
        self.column_transformer_.fit(X, y, **fit_params)

        return self

    def transform(self, X):
        check_is_fitted(self)
        return self.column_transformer_.transform(X)

    def get_feature_names_out(self, input_features=None):
        return self.column_transformer_.get_feature_names_out(input_features)
