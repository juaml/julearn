"""Implement transformer to set column types."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL

import re

import pandas as pd

from ...base import JuTransformer, change_column_type
from ...utils import logger, raise_error
from ...utils.logging import DelayedFmtMessage as __
from ...utils.typing import ColumnTypesLike, DataLike


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
        X_types: dict[str, list[str]] | None = None,  # noqa: N803
        row_select_col_type: ColumnTypesLike | None = None,
        row_select_vals: str | int | list | bool | None = None,
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
        self,
        X: pd.DataFrame,  # noqa: N803
        y: DataLike | None = None,
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
        if not isinstance(X, pd.DataFrame | pd.Series):
            X = pd.DataFrame(X)
            X.columns = X.columns.astype(str)

        self.feature_names_in_ = X.columns
        logger.debug(
            __(
                "Setting column types for {features}",
                features=self.feature_names_in_,
            )
        )

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
                if any(re.fullmatch(exp, col) for exp in columns)
            ]
            column_mapper_.update(
                {col: change_column_type(col, X_type) for col in t_columns}
            )

        logger.debug(
            __("\tColumn mappers for {mapper}", mapper=column_mapper_)
        )
        self.column_mapper_ = column_mapper_
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
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
        self, input_features: list[str] | None = None
    ) -> list[str]:
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
