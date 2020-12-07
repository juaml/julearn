import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator

from .. utils import pick_columns, change_column_type


class ChangeColumnTypes(TransformerMixin, BaseEstimator):

    def __init__(self, columns_match, new_type):
        self.columns_match = columns_match
        self.new_type = new_type

    def fit(self, X, y=None):
        self.detected_columns_ = pick_columns(self.columns_match, X.columns)
        self.column_mapper_ = {col: change_column_type(col, self.new_type)
                               for col in self.detected_columns_}
        return self

    def transform(self, X):

        return X.copy().rename(columns=self.column_mapper_)


class DropColumns(TransformerMixin, BaseEstimator):

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        self.support_mask_ = pd.Series(True, index=X.columns, dtype=bool)
        try:
            self.detected_columns_ = pick_columns(self.columns, X.columns)
            self.support_mask_[self.detected_columns_] = False
        except ValueError:
            self.detected_columns_ = []
        self.support_mask_ = self.support_mask_.values
        return self

    def transform(self, X):
        return X.drop(columns=self.detected_columns_)

    def get_support(self, indices=False):
        if indices:
            return np.arange(len(self.support_mask_))[self.support_mask_]
        else:
            return self.support_mask_
