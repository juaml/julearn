from ._pipeline import PipelineInspector
from ._cv import FoldsInspector

from ..utils.logging import raise_error


class Inspector:
    def __init__(
        self,
        scores,
        model=None,
        X=None,
        y=None,
        groups=None,
        cv=None,
    ):
        self._scores = scores
        self._model = model
        self._X = X
        self._y = y
        self._groups = groups
        self._cv = cv

    @property
    def model(self):
        if self._model is None:
            raise_error("No model was provided. Cannot inspect the model.")
        return PipelineInspector(model=self._model)

    @property
    def folds(self):
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
