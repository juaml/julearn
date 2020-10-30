from copy import deepcopy
import pandas as pd
from sklearn.base import TransformerMixin


class DataFrameTransformer(TransformerMixin):

    column_types = ['confound', 'all', 'all_feature',
                    'continuous', 'categorical']

    def __init__(self, transformer, transform_column='all',
                 returned_features='unknown', column_type_sep='__:type:__',
                 **params):
        """Core class of julearn.
        Similar to sklearns ColumnTransformer, it applies any transformer
        to a set of columns specified by transform_column.
        But the transformers are not applied independetley to each column.
        This wrapper also alwayes returns a pd.DataFrame instead of np.ndarray
        The columns of the outputted pd.DataFrame are dependent on the
        returned_features argument.

        Parameters
        ----------
        transformer : sklearn.base.TransformerMixin
            A transformer following sklearn standards.
        transform_column : str or list[str]
            This arguments decides which columns will be transformed by the
            transformer. One way is entering a list of valid column names of
            the pd.DataFrame you want to use.
            Another one is to use a valid column type.
            Column types are tagges you have to provide after the column name
            only seperated by the column_type_sep.
            Valid column types are: `confound`, 'continuous' and 'categorical'.
            Furthermore, you can enter `all` to transform all columns,
            `all_feature` to transform all columns excluding the confound
            or provide a list of valid column types.



        returned_features : str, optional
            `unknown` leads to created column names,
            `unknown_same_type` leads to created column names
             with the same column type.
            `same` leads copies the names from the original pd.DataFrame
            `subset` leads to the columns being a subset of the original
             pd.DataFrame. This functionality needs the transformer to have a
             .get_support method following sklearn standards.
             , by default `unknown`
        """

        self.transformer = transformer
        self.transform_column = transform_column
        self.returned_features = returned_features
        self.column_type_sep = column_type_sep
        self.set_params(**params)

    def fit(self, X, y=None):
        # TODO Validation missing
        self.set_columns_to_transform(X)
        X_transform = X.loc[:, self.transform_column_]
        self.transformer.fit(X_transform, y)

        return self

    def transform(self, X):

        X_transform = X.loc[:, self.transform_column_]
        X_rest = X.drop(columns=self.transform_column_)
        X_transformed = self.transformer.transform(X_transform)

        X_transformed = pd.DataFrame(X_transformed,
                                     index=X_transform.index)
        if self.returned_features == 'same':
            X_transformed.columns = X_transform.columns
            df_out = pd.concat([X_transformed, X_rest],
                               axis=1).reindex(columns=X.columns)
        elif self.returned_features == 'subset':
            if hasattr(self.transformer, 'get_support'):
                # assign feature names after transforming by using
                # sklearns way of getting the features index after transforming
                X_transformed.columns = X_transform.columns[
                    self.transformer.get_support()]
                df_out = pd.concat([X_transformed, X_rest], axis=1)
            else:
                raise ValueError('You cannot use subset on a transformer'
                                 'without a "get_support" attribute')
        elif self.returned_features == 'unknown':
            transformer_name = self.transformer.__class__.__name__.lower()
            columns = [
                f'{transformer_name}_component:{n_column}'
                f'{self.column_type_sep}continuous'
                for n_column in range(len(X_transformed.columns))]

            X_transformed.columns = columns
            df_out = pd.concat([X_transformed, X_rest], axis=1)

        elif self.returned_features == 'unknown_same_type':
            transformer_name = self.transformer.__class__.__name__.lower()
            columns = [
                f'{transformer_name}_component:{n_column}'
                f'{self.column_type_sep}{self.get_column_type(column)}'
                for n_column, column in enumerate(X_transformed.columns)]

            X_transformed.columns = columns
            df_out = pd.concat([X_transformed, X_rest], axis=1)

        else:
            raise ValueError('returned_features can only be '
                             'same, unknown, subset, '
                             f'but was {self.returned_features}')

        return df_out

    def set_columns_to_transform(self, X):

        all_columns = X.columns.copy()
        if type(self.transform_column) == str:
            if self.transform_column in self.column_types:
                self.transform_column_ = self.get_columns_of_type(
                    all_columns, self.transform_column)
            else:
                self.transform_column_ = (X.loc[:, self.transform_column]
                                          .columns.to_list())

        else:
            if (pd.Series([column in self.column_types
                           for column in self.transform_column
                           ]).all()):
                self.transform_column_ = self.get_columns_of_types(
                    X.columns, self.transform_column)
            else:
                self.transform_column_ = (X.loc[:, self.transform_column]
                                          .columns.to_list()
                                          )

        if self.transform_column_ == []:
            raise ValueError('There is not valid Column to transform '
                             f'{self.transform_column} should be selected, '
                             f'but is not valid for columns = {X.columns}'
                             )

    def initialize_params(self, transformer):
        """
        Initializes the parameters of the transformer
        for the DataFrameTransformer
        """
        transformer_params = transformer.get_params()

        for param, val in transformer_params.items():
            setattr(self, param, val)

    def get_params(self, deep=True):
        params = dict(transformer=self.transformer,
                      transform_column=self.transform_column,
                      returned_features=self.returned_features,
                      column_type_sep=self.column_type_sep)

        transformer_params = self.transformer.get_params(deep=deep)
        for param, val in transformer_params.items():
            params[param] = val
        return deepcopy(params) if deep else params

    def set_params(self, **params):
        for param in ['transformer', 'transform_column',
                      'returned_features', 'column_type_sep']:
            if params.get(param) is not None:
                setattr(self, param, params.pop(param))
        self.transformer.set_params(**params)
        # self.initialize_params(self.transformer)
        return self

    def get_support(self, indices=False):
        self.transformer.get_support(indices=indices)

    def get_column_type(self, column):
        column_type = column.split(self.column_type_sep)

        if len(column_type) != 2:
            column_type = column_type, 'continuous'

        return column_type[1]

    def get_columns_of_type(self, columns, column_type):
        if column_type == 'all':
            valid_columns = columns.to_list()
        elif column_type == 'all_features':

            valid_columns = [column
                             for column in columns
                             if self.get_column_type(column) != 'confound'
                             ]
        else:
            valid_columns = [column
                             for column in columns
                             if self.get_column_type(column) == column_type
                             ]
        return valid_columns

    def get_columns_of_types(self, columns, column_types):
        valid_columns = [valid_column
                         for column_type in column_types
                         for valid_column in self.get_columns_of_type(
                             columns, column_type)
                         ]
        return valid_columns
