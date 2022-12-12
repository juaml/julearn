from typing import TYPE_CHECKING
from sklearn.base import clone
from sklearn.utils.metaestimators import available_if
from ...base import JuBaseEstimator, _wrapped_model_has
from ...utils.typing import ModelLike


if TYPE_CHECKING:
    from ...pipeline.target_pipeline import JuTargetPipeline


class JuTransformedTargetModel(JuBaseEstimator):
    def __init__(self, model: ModelLike, transformer: "JuTargetPipeline"):
        self.model = model
        self.transformer = transformer

    def fit(self, X, y, **fit_params):
        y = self.transformer.fit_transform(X, y)
        self.model_ = clone(self.model)
        self.model_.fit(X, y, **fit_params)
        return self

    def predict(self, X):
        # TODO: Check if we can inverse the y transformations
        # Raise warning if not possible
        return self.model_.predict(X)

    @available_if(_wrapped_model_has("predict_proba"))
    def predict_proba(self, X):
        # TODO: Check if we can inverse the y transformations
        # Raise warning if not possible
        return self.model_.predict_proba(X)

    @available_if(_wrapped_model_has("decision_function"))
    def decision_function(self, X):
        return self.model_.decision_function(X)

    @property
    def classes_(self):
        return self.model_.classes_
