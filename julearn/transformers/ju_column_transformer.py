from sklearn.compose import ColumnTransformer
from sklearn.utils.validation import check_is_fitted

from .. base import JuTransformer


class JuColumnTransformer(JuTransformer):

    def __init__(self, name, transformer, apply_to,
                 needed_types=None, **params):
        self.name = name
        self.transformer = transformer
        self.apply_to = apply_to
        self.needed_types = needed_types
        self.set_params(**params)

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

    def get_params(self, deep=True):
        return dict(
            **self.transformer.get_params(True),
            name=self.name,
            apply_to=self.apply_to,
            needed_types=self.needed_types,
            transformer=self.transformer


        )

    def set_params(self, **kwargs):

        transformer_params = list(self.transformer
                                  .get_params(True)
                                  .keys()
                                  )

        for param, val in kwargs.items():
            if param in transformer_params:
                self.transformer.set_params(**{param: val})
            else:
                setattr(self, param, val)
