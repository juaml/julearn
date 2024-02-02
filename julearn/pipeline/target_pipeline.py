"""Class for pipelines that work on the target."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL

from typing import List, Tuple, Union

import numpy as np
import pandas as pd

from ..transformers.target import JuTargetTransformer
from ..utils.typing import DataLike, TransformerLike


class JuTargetPipeline:
    """Class for pipelines that work on the target.

    Unlike the :class:`sklearn.pipeline.Pipeline`, this pipeline fits and
    transforms using both X and y. This is useful for pipelines that work on
    the target but require information from the input data, such as the
    :class:`julearn.transformers.target.TargetConfoundRemover` or
    a target encoder that requires one of the features to be present.

    IMPORTANT: Using any of the transformers that transforms the target
    based on the input data will result in data leakage if the features
    are not dropped after the transformation.

    Parameters
    ----------
    steps : List[Tuple[str, Union[JuTargetTransformer, TransformerLike]]]
        List of steps to be performed on the target.

    """

    def __init__(
        self,
        steps: List[Tuple[str, Union[JuTargetTransformer, TransformerLike]]],
    ):
        if not isinstance(steps, List):
            raise TypeError("steps must be a list")
        self.steps = steps

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
        y : DataLike
            The transformed target.

        """
        return self.fit(X, y).transform(X, y)

    def fit(
        self, X: pd.DataFrame, y: DataLike  # noqa: N803
    ) -> "JuTargetPipeline":
        """Fit the target pipeline.

        Parameters
        ----------
        X : pd.DataFrame
            The input data.
        y : DataLike
            The target.

        Returns
        -------
        self : JuTargetPipeline
            The fitted pipeline.

        """
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        for _, t_step in self.steps:
            if isinstance(t_step, JuTargetTransformer):
                y = t_step.fit_transform(X, y)
            else:
                y = t_step.fit_transform(y[:, None])[:, 0]  # type: ignore
        return self

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
        y : DataLike
            The transformed target.

        """
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        for _, t_step in self.steps:
            if isinstance(t_step, JuTargetTransformer):
                y = t_step.transform(X, y)
            else:
                y = t_step.transform(y[:, None])[:, 0]  # type: ignore
        return y

    def inverse_transform(
        self, X: pd.DataFrame, y: DataLike  # noqa: N803
    ) -> DataLike:
        """Inverse transform the target.

        Parameters
        ----------
        X : pd.DataFrame
            The input data.
        y : DataLike
            The target.

        Returns
        -------
        y : DataLike
            The inverse transformed target.

        """
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        for _, t_step in reversed(self.steps):
            if isinstance(t_step, JuTargetTransformer):
                y = t_step.inverse_transform(X, y)  # type: ignore
            else:
                y = t_step.inverse_transform(y[:, None])[:, 0]  # type: ignore
        return y

    def can_inverse_transform(self) -> bool:
        """Check if the pipeline can inverse transform.

        Returns
        -------
        bool
            True if the pipeline can inverse transform.

        """
        for _, t_step in self.steps:
            if not hasattr(t_step, "inverse_transform"):
                return False
        return True

    @property
    def needed_types(self):
        """Get the needed types for the pipeline.

        Returns
        -------
        needed_types : Set of str or None
            The needed types for the pipeline.

        """
        needed_types = []
        for _, t_step in self.steps:
            if getattr(t_step, "needed_types", None) is not None:
                needed_types.extend(t_step.needed_types)
        needed_types = set(needed_types)
        return needed_types if len(needed_types) > 0 else None
