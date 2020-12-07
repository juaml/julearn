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

    def _auto_set_returned_features(self):
        """automatically set the returned_features argument based on the
        used transformer

        Parameters
        ----------
        transformer : sklearn transformer
            the transformer which will be wrapped by DataFrameTransformer
        """
        if self.returned_features is None:
            if hasattr(self.transformer, 'get_support'):
                self.returned_features = 'subset'
            else:
                self.returned_features = get_returned_features(
                    self.transformer)

    def _auto_set_apply_to(self):
        if self.apply_to is None:
            self.apply_to = get_apply_to(self.transformer)


def transform_dataframe(transformer, df, returned_features):

    df_to_trans = df.copy()
    df_trans = (pd.DataFrame(transformer.transform(df_to_trans),
                             index=df_to_trans.index)
                .rename(columns=lambda col: str(col))
                )

    valid_returned_features = [
        'same', 'subset', 'from_transformer'
        'unknown', 'unknown_same_type']

    if returned_features == 'subset':
        column_names = df_to_trans.columns[transformer.get_support()]

    elif returned_features == 'same':
        column_names = list(df_to_trans.columns)

    elif returned_features == 'from_transformer':
        column_names = list(df_trans.columns)

    elif returned_features == 'unknown_same_type':

        column_type = get_column_type(list(df.columns)[0])
        for column in df.columns:
            column_type_i = get_column_type(column)
            if column_type_i != column_type:
                raise_error(
                    "You can only use 'unknown_same_type'"
                    "if all column types are the same"
                    f"this is not for the following columns {df_trans.columns}"
                )

        transformer_name = transformer.__class__.__name__.lower()
        column_names = [
            f'{transformer_name}_component_{n}__:type:__{column_type}'
            for n, column in enumerate(df_trans.columns)
        ]

    elif returned_features is None or returned_features == 'unknown':
        transformer_name = transformer.__class__.__name__.lower()
        column_names = [
            f'{transformer_name}_component_{n}__:type:__continuous'
            for n, column in enumerate(df_trans.columns)
        ]
    else:
        raise_error(
            'There is a wrong input for returned_features. Valid inputs are: '
            f'{valid_returned_features}, but {returned_features} was used')
    df_trans.columns = column_names
    return df_trans


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
