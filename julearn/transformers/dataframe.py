import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer

from .. utils import (
    # pick_columns,
    change_column_type
)
from . base import JuTransformer
from .. utils import raise_error, logger


class SetColumnTypes(JuTransformer):

    def __init__(self, X_types=None):
        self.X_types = X_types

    def fit(self, X, y=None):
        if self.X_types is None:
            self.X_types = {}
        self.feature_names_in_ = X.columns
        logger.debug(f"Setting column types for {self.feature_names_in_}")

        # initialize the column_mapper_ using the X_types of X
        column_mapper_ = dict()
        for col in X.columns.tolist():

            if "__:type:__" in col:
                col_no_type, X_type = col.split("__:type:__")
            else:
                col_no_type, X_type = col, "continuous"
            column_mapper_[col_no_type] = change_column_type(
                col_no_type, X_type)

        for X_type, columns in self.X_types.items():
            if not isinstance(columns, (list, tuple)):
                raise_error(
                    "Each value of X_types must be either a list or a tuple.")
            column_mapper_.update({
                col: change_column_type(col, X_type)
                for col in columns})

        logger.debug(f"\tColumn mappers for {column_mapper_}")
        self.column_mapper_ = column_mapper_
        return self

    def transform(self, X):
        return (X
                # remove column types of input
                .rename(columns=lambda col: col.split("__:type:__")[0])
                # assgin new column types (previous as default)
                .rename(columns=self.column_mapper_)
                )

    def get_feature_names_out(self, input_features=None):
        return (self.feature_names_in_
                .map(lambda col: col.split("__:type:__")[0])
                .map(self.column_mapper_)
                .values
                )


class FilterColumns(JuTransformer):

    def __init__(self, apply_to=None, keep=None, needed_types=None):
        self.apply_to = apply_to
        self.keep = keep
        self.needed_types = needed_types

    def fit(self, X, y=None):
        self.apply_to = (
            "continuous" if self.apply_to is None else self.apply_to)
        self.keep = "continuous" if self.keep is None else self.keep
        self.apply_to = self._ensure_apply_to()
        self.keep = self._ensure_column_types(self.keep)
        self.needed_types = self._ensure_needed_types()

        inner_selector = self.apply_to.to_type_selector()
        inner_filter = ColumnTransformer(
            transformers=[
                ("filter_apply_to", "passthrough", inner_selector), ],
            remainder="passthrough", verbose_feature_names_out=False,
        )

        apply_to_selector = self.keep.to_type_selector()
        self.filter_columns_ = ColumnTransformer(
            transformers=[
                ("keep", inner_filter, apply_to_selector)],
            remainder="drop", verbose_feature_names_out=False,

        )
        self.filter_columns_.fit(X, y)
        return self

    def transform(self, X):
        return self.filter_columns_.transform(X)

    def get_feature_names_out(self, input_features=None):
        return self.filter_columns_.get_feature_names_out(input_features)


class ChangeColumnTypes(JuTransformer):

    def __init__(self, apply_to, X_types_renamer, needed_types=None, ):
        self.apply_to = apply_to
        self.needed_types = needed_types
        self.X_types_renamer = X_types_renamer

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return (
            self.fitler_columns(X)
            .rename(columns=self.X_types_renamer)
        )


class DropColumns(JuTransformer):

    def __init__(self, apply_to, needed_types=None):
        self.apply_to = apply_to
        self.needed_types = needed_types

    def fit(self, X, y=None):
        self.apply_to = self._ensure_apply_to()
        self.support_mask_ = pd.Series(True, index=X.columns, dtype=bool)

        try:
            self.drop_columns_ = self.filter_columns(X).columns
            self.support_mask_[self.drop_columns_] = False
        except ValueError:
            self.drop_columns_ = []
        self.support_mask_ = self.support_mask_.values
        return self

    def transform(self, X):
        return X.drop(columns=self.drop_columns_)

    def get_support(self, indices=False):
        if indices:
            return np.arange(len(self.support_mask_))[self.support_mask_]
        else:
            return self.support_mask_
