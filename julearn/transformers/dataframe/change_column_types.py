"""Implement transformer to change column types."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL

from typing import Dict, Optional, List

import pandas as pd

from ...base import JuTransformer, ColumnTypesLike
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
    """

    def __init__(
        self,
        X_types_renamer: Dict[str, str],
        apply_to: ColumnTypesLike,
    ):
        self.X_types_renamer = X_types_renamer
        super().__init__(apply_to=apply_to, needed_types=None)

    def fit(
        self, X: pd.DataFrame, y: Optional[DataLike] = None
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

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
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
        out = out.map(self._renamer)
        return out  # type: ignore
