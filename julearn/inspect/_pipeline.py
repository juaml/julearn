import re

from sklearn.utils.validation import check_is_fitted

from ..transformers import JuColumnTransformer


class PipelineInspector():

    def __init__(self, model):
        check_is_fitted(model)
        self._model = model

    def get_step_names(self):
        return list(self._model.named_steps.keys())

    def get_step(self, name, as_estimator=False):
        step = self._model.named_steps[name]
        if not as_estimator:
            step = _EstimatorInspector(step)
        return step

    def get_params(self):

        if hasattr(self._model, "best_estimator_"):
            self._model.best_estimator_.get_params()
        return self._model.get_params()

    def get_fitted_params(self):
        fitted_params = {}
        model = (self._model.best_estimator_
                 if hasattr(self._model, "best_estimator_")
                 else self._model
                 )
        for name, step in model.steps:
            params = _EstimatorInspector(step).get_fitted_params()
            fitted_params = {
                **fitted_params,
                ** {f"{name}__{param}": val for param, val in params.items()}
            }
        return fitted_params


class _EstimatorInspector():
    def __init__(self, estimator):
        self._estimator = estimator

    def get_params(self):
        return self._estimator.get_params()

    def get_fitted_params(self):
        all_params = vars(self._estimator)
        if isinstance(self._estimator, JuColumnTransformer):
            all_params = {
                **all_params,
                **vars(self._estimator.column_transformer_.transformers_[0][1])
            }

        return {param: val for param, val in all_params.items()
                if re.match(r"^[a-zA-Z].*[a-zA-Z0-9]*_$", param)
                }

    @property
    def estimator(self):
        return self._estimator
