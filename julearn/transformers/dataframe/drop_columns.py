"""Implement transformer to change drop columns."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL

from typing import Optional, Union

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from ...base import ColumnTypesLike, JuTransformer
from ...utils import logger
from ...utils.typing import DataLike


class DropColumns(JuTransformer):
    """Drop columns of a DataFrame.

    Parameters
    ----------
    apply_to : ColumnTypesLike
        The feature types ('X_types') to drop.
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
        apply_to: ColumnTypesLike,
        row_select_col_type: Optional[ColumnTypesLike] = None,
        row_select_vals: Optional[Union[str, int, list, bool]] = None,
    ):
        super().__init__(
            apply_to=apply_to,
            needed_types=None,
            row_select_col_type=row_select_col_type,
            row_select_vals=row_select_vals,
        )

    def _fit(
        self, X: pd.DataFrame, y: Optional[DataLike] = None  # noqa: N803
    ) -> "DropColumns":
        """Fit the transformer.

        The transformer will learn how to drop the columns of the input data.

        Parameters
        ----------
        X : pd.DataFrame
            Data to drop columns.
        y : Data-Like, optional
            Target data. This data will not be used.

        Returns
        -------
        DropColumns
            The fitted transformer.

        """
        self.support_mask_ = pd.Series(True, index=X.columns, dtype=bool)

        try:
            self.drop_columns_ = self.filter_columns(X).columns
            self.support_mask_[self.drop_columns_] = False
        except ValueError:
            self.drop_columns_ = []
        self.support_mask_ = self.support_mask_.values
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        """Drop the columns.

        Parameters
        ----------
        X : pd.DataFrame
            Data to drop columns.

        Returns
        -------
        pd.DataFrame
            Data with dropped columns.

        """
        logger.debug(f"Dropping columns: {self.drop_columns_}")
        return X.drop(columns=self.drop_columns_)

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
        support_mask : numpy.array
            The support mask

        """
        if indices:
            return np.arange(len(self.support_mask_))[
                self.support_mask_
            ]  # type: ignore
        else:
            return self.support_mask_  # type: ignore
