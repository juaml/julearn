"""Base class for target transformers."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL

import pandas as pd

from ...utils.typing import DataLike


class JuTargetTransformer:
    """Base class for target transformers.

    Unlike the scikit-learn transformer, this fits and transforms using both
    X and y. This is useful for pipelines that work on the target but require
    information from the input data, such as the TargetConfoundRemover or
    a target encoder that requires one of the features to be present.

    IMPORTANT: Using any of the transformers that transforms the target
    based on the input data will result in data leakage if the features
    are not dropped after the transformation.
    """

    def fit_transform(
        self, X: pd.DataFrame, y: DataLike  # noqa: N803
    ) -> DataLike:
        """Fit and transform the target.

        Parameters
        ----------
        X : pd.DataFrame
            The input data.
        y : DataLike
            The target.

        Returns
        -------
        DataLike
            The transformed target.

        """
        return self.fit(X, y).transform(X, y)

    def fit(
        self, X: pd.DataFrame, y: DataLike  # noqa: N803
    ) -> "JuTargetTransformer":
        """Fit the target.

        Parameters
        ----------
        X : pd.DataFrame
            The input data.
        y : DataLike
            The target.

        Returns
        -------
        JuTargetTransformer
            The fitted transformer.

        """
        raise NotImplementedError("fit() not implemented")

    def transform(
        self, X: pd.DataFrame, y: DataLike  # noqa: N803
    ) -> DataLike:
        """Transform the target.

        Parameters
        ----------
        X : pd.DataFrame
            The input data.
        y : DataLike
            The target.

        Returns
        -------
        DataLike
            The transformed target.

        """
        raise NotImplementedError("transform() not implemented")
