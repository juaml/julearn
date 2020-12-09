# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL
from copy import deepcopy
import pandas as pd
from sklearn.base import TransformerMixin

from .. utils import (raise_error,
                      get_column_type)

from . available_transformers import _get_returned_features, _get_apply_to

# rename file to meta.py

# DataFrameWrapTransformer


class DataFrameWrapTransformer(TransformerMixin):

    column_types = ['confound', 'all', 'all_features',
                    'continuous', 'categorical']

    column_type_sep = '__:type:__'

    def __init__(self, transformer, apply_to=None,
                 returned_features=None,
                 **params):
        """Transformer to wrap other transformers and apply and return a
        DataFrame. Allows to apply the original transformer to only some of the
        columns from the DataFrame, while preserving other columns as they
        are. The returned output will be named using the information provided
        by the 'returned_features' argument and the inputted DataFrame

        Parameters
        ----------
        transformer : object
            A transformer following sklearn standards.

        apply_to : str | list(str) or None
            Defines to which columns the transformer is applied to.
            For this julearn user specified 'columns_types' from the user.
            All other columns will be ignored by the transformer and kept as
            they are.
            apply_to can be set to one or multiple of the following options:

                * 'all': The transformer is applied to all columns
                * 'all_features': The transformer is applied to continuous and
                    categorical features.
                * 'continuous': The transformer is only applied to continuous
                    features.
                * 'categorical': The transformer is only applied to categorical
                    features.
                * 'confound': The transformer is only applied to confounds.

            As mentioned above you can combine these types.
            E.g. ['continuous', 'confound'] would specify that your transformer
            uses both the confounds and the continuous variables as input.

        returned_features : str | None
            helps to name the transformed DataFrame accordingly.
            Using one of the following strings will lead to the noted behavior:

            * 'None': here returned_features will be set automatically using
                information which is registered in julearn.
                This should also be considered as the default option for any
                user and only set otherwise in very specific use cases

            * 'same': leads copies the names from the original pd.DataFrame
            * 'subset': leads to the columns being a subset of the original
              pd.DataFrame. This functionality needs the transformer to have a
              .get_support method following sklearn standards.
            * 'from_transformer': the outputted columns are already defined in
              the transformer
            * 'unknown' leads to created column names,
            * 'unknown_same_type' leads to created column names
              with the same column type.

        """

        self.transformer = transformer
        self.apply_to = apply_to
        self.returned_features = returned_features
        self.set_params(**params)
        self._check_apply_to_returned_features()

    def fit(self, X, y=None):
        self._auto_set_returned_features()
        self._auto_set_apply_to()
        self._set_columns_to_transform(X)
        X_transform = X.loc[:, self.transform_column_]
        self.transformer.fit(X_transform, y)

        return self

    def transform(self, X):

        X_to_transform = X.loc[:, self.transform_column_]
        X_rest = X.drop(columns=self.transform_column_)

        X_transformed = transform_dataframe(
            self.transformer, X_to_transform,
            returned_features=self.returned_features)

        df_out = pd.concat([X_transformed, X_rest], axis=1)

        if df_out.columns.isin(X.columns).all():
            columns_ordered = [col
                               for col in X.columns
                               if col in df_out.columns]
            df_out = df_out.reindex(columns=columns_ordered)
        return df_out

    def get_params(self, deep=True):
        params = dict(transformer=self.transformer,
                      apply_to=self.apply_to,
                      returned_features=self.returned_features,
                      )

        transformer_params = self.transformer.get_params(deep=deep)
        for param, val in transformer_params.items():
            params[param] = val
        return deepcopy(params) if deep else params

    def set_params(self, **params):
        for param in ['transformer', 'apply_to',
                      'returned_features']:
            if params.get(param) is not None:
                setattr(self, param, params.pop(param))
        self.transformer.set_params(**params)
        return self

    def get_support(self, indices=False):
        self.transformer.get_support(indices=indices)

    def __repr__(self):
        return (f"DataFrameWrapTransformer({self.transformer.__repr__()}, "
                f"apply_to='{self.apply_to}', "
                f"return_features='{self.returned_features}'"
                )

    def _set_columns_to_transform(self, X):

        all_columns = X.columns.copy()
        if type(self.apply_to) == str:
            if self.apply_to in self.column_types:
                self.transform_column_ = self._get_columns_of_type(
                    all_columns, self.apply_to)
            else:
                self.transform_column_ = (X.loc[:, self.apply_to]
                                          .columns.to_list())

        else:
            if (pd.Series([column in self.column_types
                           for column in self.apply_to
                           ]).all()):
                self.transform_column_ = self._get_columns_of_types(
                    X.columns, self.apply_to)
            else:
                self.transform_column_ = (X.loc[:, self.apply_to]
                                          .columns.to_list()
                                          )

        if self.transform_column_ == []:
            raise_error('There is not valid column to transform '
                        f'{self.apply_to} should be selected, '
                        f'but is not valid for columns = {X.columns}')

    def _get_column_type(self, column):
        column_type = column.split(self.column_type_sep)

        if len(column_type) != 2:
            column_type = column_type, 'continuous'

        return column_type[1]

    def _get_columns_of_type(self, columns, column_type):
        if column_type == 'all':
            valid_columns = columns.to_list()
        elif column_type == 'all_features':
            valid_columns = [column
                             for column in columns
                             if self._get_column_type(column) != 'confound'
                             ]

        else:
            valid_columns = [column
                             for column in columns
                             if self._get_column_type(column) == column_type
                             ]
        return valid_columns

    def _get_columns_of_types(self, columns, column_types):
        valid_columns = [valid_column
                         for column_type in column_types
                         for valid_column in self._get_columns_of_type(
                             columns, column_type)
                         ]
        return valid_columns

    def _auto_set_returned_features(self):
        """automatically set the returned_features argument based on the
        used transformer

        Parameters
        ----------
        transformer : sklearn transformer
            the transformer which will be wrapped by DataFrameWrapTransformer
        """
        if self.returned_features is None:
            self.returned_features = _get_returned_features(
                self.transformer)

            if (self.apply_to == 'confound') and (
                    self.returned_features == 'unknown'):
                self.returned_features = 'unknown_same_type'

    def _auto_set_apply_to(self):
        if self.apply_to is None:
            self.apply_to = _get_apply_to(self.transformer)

    def _check_apply_to_returned_features(self):
        _get_apply_to(self.transformer)
        _get_returned_features(self.transformer)


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
        if hasattr(transformer, 'get_support'):
            column_names = df_to_trans.columns[transformer.get_support()]
        else:
            raise_error(
                'You can only subset with a transformer which has '
                f'a `get_support` method. {transformer} does not have a'
                '`get_support` method. Most probably you have registered your '
                'transformer wrongly or set the hyperparameter '
                '`return_features` wrongly.')

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
