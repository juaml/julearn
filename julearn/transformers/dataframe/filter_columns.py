"""Implement transformer to filter columns."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL

from typing import List, Optional, Union

import pandas as pd
from sklearn.compose import ColumnTransformer

from ...base import (
    ColumnTypes,
    ColumnTypesLike,
    JuTransformer,
    ensure_column_types,
)
from ...utils.typing import DataLike


class FilterColumns(JuTransformer):
    """Filter columns of a DataFrame.

    Parameters
    ----------
    keep : ColumnTypesLike, optional
        Which feature types ('X_types') to keep. If not specified, 'keep'
        defaults to 'continuous'.
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
        keep: Optional[ColumnTypesLike] = None,
        row_select_col_type:  Optional[ColumnTypesLike] = None,
        row_select_vals:  Optional[Union[str,
                                         int, list, bool]] = None,
    ):
        if keep is None:
            keep = "continuous"
        self.keep: ColumnTypes = ensure_column_types(keep)
        super().__init__(
            apply_to="*", needed_types=keep,
            row_select_col_type=row_select_col_type,
            row_select_vals=row_select_vals

        )

    def _fit(
        self, X: pd.DataFrame, y: Optional[DataLike] = None
    ) -> "FilterColumns":
        """Fit the transformer.

        Parameters
        ----------
        X : pd.DataFrame
            The data to fit the transformer on.
        y : DataLike, optional
            The target data. This data will not be used.

        Returns
        -------
        FilterColumns
            The fitted transformer.
        """
        apply_to_selector = self.keep.to_type_selector()
        self.filter_columns_ = ColumnTransformer(
            transformers=[("keep", "passthrough", apply_to_selector)],
            remainder="drop",
            verbose_feature_names_out=False,
        )
        self.filter_columns_.fit(X, y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the data.

        Parameters
        ----------
        X : pd.DataFrame
            The data to filter the columns on.

        Returns
        -------
        DataLike
            The filtered data.
        """
        return self.filter_columns_.transform(X)  # type: ignore

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
        out = self.filter_columns_.get_feature_names_out(input_features)
        return out  # type: ignore
