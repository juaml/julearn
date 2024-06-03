"""Implement transformer to change column types."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL

from typing import Dict, List, Optional, Union

import pandas as pd

from ...base import ColumnTypesLike, JuTransformer
from ...utils.typing import DataLike


class ChangeColumnTypes(JuTransformer):
    """Transformer to change the column types.

    Parameters
    ----------
    X_types : dict, optional
        A dictionary with the column types to set. The keys are the column
        types and the values are the columns to set the type to. If None, will
        set all the column types to `continuous` (default is None).
    apply_to : ColumnTypesLike, optional
        From which feature types ('X_types') to remove confounds.
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
        X_types_renamer: Dict[str, str],  # noqa: N803
        apply_to: ColumnTypesLike,
        row_select_col_type: Optional[ColumnTypesLike] = None,
        row_select_vals: Optional[Union[str, int, list, bool]] = None,
    ):
        self.X_types_renamer = X_types_renamer
        super().__init__(
            apply_to=apply_to,
            needed_types=None,
            row_select_col_type=row_select_col_type,
            row_select_vals=row_select_vals,
        )

    def _fit(
        self, X: pd.DataFrame, y: Optional[DataLike] = None  # noqa: N803
    ) -> "ChangeColumnTypes":
        """Fit the transformer.

        The transformer will learn how to se set the column types of the input
        data. This will not transform the data yet.

        Parameters
        ----------
        X : pd.DataFrame
            Data to add column types.
        y : Data-Like, optional
            Target data. This data will not be used.

        Returns
        -------
        ChangeColumnTypes
            The fitted transformer.

        """
        self.feature_names_in_ = X.columns
        to_rename = {}
        for col in self.filter_columns(X).columns.tolist():
            if "__:type:__" in col:
                name, old_type = col.split("__:type:__")
                if old_type in self.X_types_renamer:
                    to_rename[
                        col
                    ] = f"{name}__:type:__{self.X_types_renamer[old_type]}"
        self._renamer = to_rename
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        """Change the column types.

        Parameters
        ----------
        X : pd.DataFrame
            Data to set the column types.

        Returns
        -------
        pd.DataFrame
            The transformed data.

        """
        return X.rename(columns=self._renamer)

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
        out = self.filter_columns(pd.DataFrame(columns=out)).columns
        # Assign new column types (previous as default)
        out = out.map(self._renamer)
        return out  # type: ignore
