from typing import (
    runtime_checkable,
    Protocol,
    Optional,
    Any,
    Dict,
    Union,
    List,
    Set,
)
import numpy as np
import pandas as pd

from ..base import ColumnTypes


DataLike = Union[np.ndarray, pd.DataFrame, pd.Series]
ColumnTypesLike = Union[List[str], Set[str], str, ColumnTypes]


@runtime_checkable
class EstimatorLike(Protocol):
    def fit(self, X, y, **kwargs: Any) -> "EstimatorLike":
        return self

    def get_params(self, deep=True) -> Dict:
        return {}

    def set_params(self, **params) -> "EstimatorLike":
        return self


@runtime_checkable
class TransformerLike(EstimatorLike, Protocol):
    def fit(self, X, y=None, **fit_params):
        pass

    def transform(self, X: DataLike) -> DataLike:
        return X

    def fit_transform(
        self, X: DataLike, y: Optional[DataLike] = None
    ) -> DataLike:
        return X


@runtime_checkable
class ModelLike(EstimatorLike, Protocol):
    classes_: np.ndarray
    def predict(self, X) -> DataLike:
        return np.zeros(1)

    def score(self, X, y, sample_weight=None) -> float:
        return 0.0

@runtime_checkable
class JuEstimatorLike(EstimatorLike, Protocol):
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
