"""Implement transformer to pick columns by name."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
# License: AGPL

from typing import Optional, Union

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.base import check_is_fitted

from ...base import (
    ColumnTypesLike,
    JuTransformer,
)
from ...utils import logger, raise_error
from ...utils.typing import DataLike


class PickColumns(JuTransformer):
    """Pick columns of a DataFrame by name.

    Parameters
    ----------
    keep : str
        Which feature (names) to keep.
    row_select_col_type : str or list of str or set of str or \
        ColumnTypes, optional
            The column types needed to select rows (default is None)
            Not really useful for this one, but here for compatibility.
    row_select_vals : str, int, bool or list of str, int, bool, optional
        The value(s) which should be selected in the row_select_col_type
        to select the rows used for training (default is None)
        Not really useful for this one, but here for compatibility.

    """

    def __init__(
        self,
        keep: str,
        row_select_col_type: Optional[ColumnTypesLike] = None,
        row_select_vals: Optional[Union[str, int, list, bool]] = None,
    ) -> None:
        self.keep = keep
        super().__init__(
            apply_to="*",
            needed_types=None,
            row_select_col_type=row_select_col_type,
            row_select_vals=row_select_vals,
        )

    def _fit(
        self,
        X: pd.DataFrame,  # noqa: N803
        y: Optional[DataLike] = None,
    ) -> "PickColumns":
        """Fit the transformer.

        Parameters
        ----------
        X : pd.DataFrame
            The data to fit the transformer on.
        y : DataLike or None, optional
            The target data. This data will not be used (default is None).

        Returns
        -------
        PickColumns
            The fitted transformer.

        """
        self.feature_names_in_ = X.columns
        self.support_mask_ = pd.Series(False, index=X.columns, dtype=bool)

        self.keep_columns_ = self.filter_columns(X).columns
        to_keep = self.keep
        if not isinstance(to_keep, list):
            to_keep = [to_keep]
        self.keep_columns_ = [
            col for col in self.keep_columns_ if col in to_keep
        ]
        if len(self.keep_columns_) != len(to_keep):
            raise_error(
                f"Kept columns do not match. Selected: ({self.keep}). "
                f"Options are: {X.columns}"
            )
        self.support_mask_[self.keep_columns_] = True
        self.support_mask_ = self.support_mask_.values
        return self

    def transform(
        self,
        X: pd.DataFrame,  # noqa: N803
    ) -> Union[pd.DataFrame, pd.Series]:
        """Pick the columns.

        Parameters
        ----------
        X : pd.DataFrame
            Data to drop columns.

        Returns
        -------
        pd.DataFrame or pd.Series
            Data with dropped columns.

        """
        logger.debug(f"Picking columns: {self.keep_columns_}")
        if len(self.keep_columns_) == 1:
            out = X[self.keep_columns_[0]]
        else:
            out = X[self.keep_columns_]
        return out

    def get_support(
        self, indices: bool = False
    ) -> Union[ArrayLike, pd.Series]:
        """Get the support mask.

        Parameters
        ----------
        indices : bool
            If true, return indices.

        Returns
        -------
        support_mask : numpy.array or pd.Series
            The support mask

        """
        if indices:
            return np.arange(len(self.support_mask_))[self.support_mask_]
        else:
            return self.support_mask_  # type: ignore

    def get_feature_names_out(
        self, input_features: Optional[list[str]] = None
    ) -> list[str]:
        """Get output feature names.

        Parameters
        ----------
        input_features : list of str, optional
            Input feature names (not used).

        Returns
        -------
        output_feature_names : list of str
            Output feature names.

        """
        check_is_fitted(self)
        out = self.feature_names_in_  # type: ignore
        return out[self.support_mask_]
