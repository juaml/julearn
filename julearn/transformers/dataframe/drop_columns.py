"""Implement transformer to change drop columns."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL

from typing import Optional, Union

import numpy as np
from numpy.typing import ArrayLike
import pandas as pd

from ...base import JuTransformer, ColumnTypesLike
from ...utils import logger
from ...utils.typing import DataLike


class DropColumns(JuTransformer):
    """Drop columns of a DataFrame.

    Parameters
    ----------
    apply_to : ColumnTypesLike
        The feature types ('X_types') to drop.
    """

    def __init__(self, apply_to: ColumnTypesLike):
        super().__init__(apply_to=apply_to, needed_types=None)

    def fit(
        self, X: pd.DataFrame, y: Optional[DataLike] = None
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

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
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
