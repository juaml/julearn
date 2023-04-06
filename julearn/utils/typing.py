"""Protocols for type checking."""

from typing import Any, Dict, Optional, Protocol, Union, runtime_checkable

import numpy as np
import pandas as pd
from sklearn.metrics._scorer import (
    _PredictScorer,
    _ProbaScorer,
    _ThresholdScorer,
)

from ..base import ColumnTypes


DataLike = Union[np.ndarray, pd.DataFrame, pd.Series]

ScorerLike = Union[_ProbaScorer, _ThresholdScorer, _PredictScorer]


@runtime_checkable
class EstimatorLikeFit1(Protocol):
    def fit(self, X, y, **kwargs: Any) -> "EstimatorLikeFit1":
        return self

    def get_params(self, deep=True) -> Dict:
        return {}

    def set_params(self, **params) -> "EstimatorLikeFit1":
        return self


@runtime_checkable
class EstimatorLikeFit2(Protocol):
    def fit(self, X, y) -> "EstimatorLikeFit2":
        return self

    def get_params(self, deep=True) -> Dict:
        return {}

    def set_params(self, **params) -> "EstimatorLikeFit2":
        return self


@runtime_checkable
class EstimatorLikeFity(Protocol):
    def fit(self, y) -> "EstimatorLikeFity":
        return self

    def get_params(self, deep=True) -> Dict:
        return {}

    def set_params(self, **params) -> "EstimatorLikeFity":
        return self


EstimatorLike = Union[EstimatorLikeFit1, EstimatorLikeFit2, EstimatorLikeFity]


@runtime_checkable
class TransformerLike(EstimatorLikeFit1, Protocol):
    def fit(self, X, y=None, **fit_params):
        pass

    def transform(self, X: DataLike) -> DataLike:
        return X

    def fit_transform(
        self, X: DataLike, y: Optional[DataLike] = None
    ) -> DataLike:
        return X


@runtime_checkable
class ModelLike(EstimatorLikeFit1, Protocol):
    classes_: np.ndarray

    def predict(self, X) -> DataLike:
        return np.zeros(1)

    def score(self, X, y, sample_weight=None) -> float:
        return 0.0


@runtime_checkable
class JuEstimatorLike(EstimatorLikeFit1, Protocol):
    def get_needed_types(self) -> ColumnTypes:
        return ColumnTypes("placeholder")

    def get_apply_to(self) -> ColumnTypes:
        return ColumnTypes("placeholder")


@runtime_checkable
class JuModelLike(ModelLike, Protocol):
    def get_needed_types(self) -> ColumnTypes:
        return ColumnTypes("placeholder")

    def get_apply_to(self) -> ColumnTypes:
        return ColumnTypes("placeholder")
