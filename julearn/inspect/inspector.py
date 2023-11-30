"""Provide base class for inspector."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL

from typing import TYPE_CHECKING, List, Optional, Union

from ..utils.logging import raise_error
from ._cv import FoldsInspector
from ._pipeline import PipelineInspector


if TYPE_CHECKING:
    import pandas as pd
    from sklearn.base import BaseEstimator

    from ..pipeline.pipeline_creator import PipelineCreator


class Inspector:
    """Base class for inspector.

    Parameters
    ----------
    scores : pd.DataFrame
        The scores as dataframe.
    model : str, optional
        The model to inspect (default None).
    X : list of str, optional
        The features as list (default None).
    y : str, optional
        The target (default None).
    groups : str, optional
        The grouping labels in case a group CV is used (default None).
    cv : int, optional
        The number of folds for cross-validation (default None).

    """

    def __init__(
        self,
        scores: "pd.DataFrame",
        model: Union[
            str,
            "PipelineCreator",
            List["PipelineCreator"],
            "BaseEstimator",
            None,
        ] = None,
        X: Optional[List[str]] = None,  # noqa: N803
        y: Optional[str] = None,
        groups: Optional[str] = None,
        cv: Optional[int] = None,
    ) -> None:
        self._scores = scores
        self._model = model
        self._X = X
        self._y = y
        self._groups = groups
        self._cv = cv

    @property
    def model(self) -> PipelineInspector:
        """Return the model.

        Returns
        -------
        PipelineInspector
            A PipelineInspector instance with model set.

        Raises
        ------
        ValueError
            If no ``model`` is provided.

        """
        if self._model is None:
            raise_error("No model was provided. Cannot inspect the model.")
        return PipelineInspector(model=self._model)

    @property
    def folds(self) -> FoldsInspector:
        """Return the folds.

        Returns
        -------
        FoldsInspector
            A FoldsInspector instance with parameters set.

        Raises
        ------
        ValueError
            If no ``cv``, ``X`` or ``y`` is provided.

        """
        if self._cv is None:
            raise_error("No cv was provided. Cannot inspect the folds.")
        if self._X is None:
            raise_error("No X was provided. Cannot inspect the folds.")
        if self._y is None:
            raise_error("No y was provided. Cannot inspect the folds.")

        return FoldsInspector(
            scores=self._scores,
            X=self._X,
            y=self._y,
            groups=self._groups,
            cv=self._cv,
        )
