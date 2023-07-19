"""Implement transformer to set column types."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL

import re
from typing import Dict, List, Optional, Union

import pandas as pd

from ...base import ColumnTypesLike, JuTransformer, change_column_type
from ...utils import logger, raise_error
from ...utils.typing import DataLike


class SetColumnTypes(JuTransformer):
    """Transformer to set the column types.

    Parameters
    ----------
    X_types : dict, optional
        A dictionary with the column types to set. The keys are the column
        types and the values are the columns to set the type to. If None, will
        set all the column types to `continuous` (default is None).
    row_select_col_type : str or list of str or set of str or ColumnTypes
        The column types needed to select rows (default is None)
        Not really useful for this one, but here for compatibility.
    row_select_vals : str, int, bool or list of str, int, bool
        The value(s) which should be selected in the row_select_col_type
        to select the rows used for training (default is None)
        Not really useful for this one, but here for compatibility.
    """

    def __init__(
        self,
        X_types: Optional[Dict[str, List[str]]] = None,
        row_select_col_type: Optional[ColumnTypesLike] = None,
        row_select_vals: Optional[Union[str, int, list, bool]] = None,
    ):
        if X_types is None:
            X_types = {}

        for X_type, columns in X_types.items():
            if not isinstance(columns, list):
                raise_error(
                    "Each value of X_types must be a list. "
                    f"Found {X_type} with value {columns} "
                    f"of type {type(columns)}"
                )
        self.X_types = X_types
        super().__init__(
            apply_to="*",
            needed_types=None,
            row_select_col_type=row_select_col_type,
            row_select_vals=row_select_vals,
        )

    def _fit(
        self, X: pd.DataFrame, y: Optional[DataLike] = None
    ) -> "SetColumnTypes":
        """Fit the transformer.

        The transformer will learn how to se set the column types of the input
        data. This will not transform the data yet.

        Parameters
        ----------
        X : pd.DataFrame
            Data to add column types.
        y : Data-Like, optional
            Target data. This data remains unchanged.

        Returns
        -------
        SetColumnTypes
            The fitted transformer.
        """
        if not isinstance(X, (pd.DataFrame, pd.Series)):
            X = pd.DataFrame(X)
            X.columns = X.columns.astype(str)

        self.feature_names_in_ = X.columns
        logger.debug(f"Setting column types for {self.feature_names_in_}")

        # initialize the column_mapper_ using the X_types of X
        column_mapper_ = {}
        for col in X.columns.tolist():
            if "__:type:__" in col:
                col_no_type, X_type = col.split("__:type:__")
            else:
                col_no_type, X_type = col, "continuous"
            column_mapper_[col_no_type] = change_column_type(
                col_no_type, X_type
            )

        # Now update the column_mapper_ with the X_types of self
        for X_type, columns in self.X_types.items():
            t_columns = [
                col
                for col in X.columns
                if any([re.fullmatch(exp, col) for exp in columns])
            ]
            column_mapper_.update(
                {col: change_column_type(col, X_type) for col in t_columns}
            )

        logger.debug(f"\tColumn mappers for {column_mapper_}")
        self.column_mapper_ = column_mapper_
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the data.

        Parameters
        ----------
        X : pd.DataFrame
            Data to set the column types.

        Returns
        -------
        pd.DataFrame
            The same dataframe.
        """
        return X

    def get_feature_names_out(
        self, input_features: Optional[List[str]] = None
    ) -> List[str]:
        """Get names of features to be returned.

        Parameters
        ----------
        input_features : None
            Parameter to ensure scikit-learn compatibility. It is not used by
            the method.

        Returns
        -------
        list
            Names of features to be kept in the output pd.DataFrame.
        """
        out = self.feature_names_in_
        # Remove column types of input
        out = out.map(lambda col: col.split("__:type:__")[0])
        # Assign new column types (previous as default)
        out = out.map(self.column_mapper_)
        return out  # type: ignore
