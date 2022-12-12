"""Implement transformer to filter columns."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL

from typing import List, Optional

import pandas as pd
from sklearn.compose import ColumnTransformer

from ...base import (
    JuTransformer,
    ensure_column_types,
    ColumnTypesLike,
    ColumnTypes,
)
from ...utils.typing import DataLike


class FilterColumns(JuTransformer):
    """Filter columns of a DataFrame.

    Parameters
    ----------
    keep : ColumnTypesLike, optional
        Which feature types ('X_types') to keep. If not specified, 'keep'
        defaults to 'continuous'.
    """

    def __init__(
        self,
        keep: Optional[ColumnTypesLike] = None,
    ):
        if keep is None:
            keep = "continuous"
        self.keep: ColumnTypes = ensure_column_types(keep)
        super().__init__(apply_to="*", needed_types=keep)

    def fit(
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
