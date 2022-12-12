"""Class for pipelines that work on the target."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL

from typing import List, Tuple, Union

import numpy as np
import pandas as pd

from ..transformers.target import JuTargetTransformer
from ..utils.typing import TransformerLike, DataLike


class JuTargetPipeline:
    """Class for pipelines that work on the target.

    Unlike the sklearn pipeline, this pipeline fits and tranforms using both
    X and y. This is useful for pipelines that work on the target but require
    information from the input data, such as the TargetConfoundRemover or
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
        self.steps = steps

    def fit_transform(self, X: pd.DataFrame, y: DataLike) -> DataLike:
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

    def fit(self, X: pd.DataFrame, y: DataLike) -> "JuTargetPipeline":
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
                y = t_step.fit_transform(y[:, None])[:, 0]
        return self

    def transform(self, X: pd.DataFrame, y: DataLike) -> DataLike:
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
                y = t_step.transform(y[:, None])[:, 0]
        return y
