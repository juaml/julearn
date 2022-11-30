from typing import runtime_checkable, Protocol


@runtime_checkable
class EstimatorLike(Protocol):
    def fit(self, X, y=None):
        pass

    def get_params(self, deep=True):
        pass

    def set_params(self, **set_params):
        pass


@runtime_checkable
class TransformerLike(EstimatorLike, Protocol):
    def transform(self, X):
        pass

    def fit_transform(self, X, y=None):
        pass


@runtime_checkable
class ModelLike(EstimatorLike, Protocol):
    def predict(self, X):
        pass


@runtime_checkable
class JuEstiamtorLike(EstimatorLike, Protocol):
    def get_needed_types(self):
        return self.needed_types

    def get_apply_to(self):
        return self.apply_to


@runtime_checkable
class JuModelLike(ModelLike, Protocol):
    def get_needed_types(self):
        return self.needed_types

    def get_apply_to(self):
        return self.apply_to
