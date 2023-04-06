import re

from sklearn.utils.validation import check_is_fitted

from ..transformers import JuColumnTransformer


class PipelineInspector():

    def __init__(self, pipe):
        check_is_fitted(pipe)
        self._pipe = pipe

    def get_step_names(self):
        return list(self._pipe.named_steps.keys())

    def get_step(self, name, as_estimator=False):
        step = self._pipe.named_steps[name]
        if not as_estimator:
            step = EstimatorInspector(step)
        return step

    def get_params(self):
        return self._pipe.get_params()

    def get_fitted_params(self):
        fitted_params = {}
        for name, step in self._pipe.steps:
            params = EstimatorInspector(step).get_fitted_params()
            fitted_params = {
                **fitted_params,
                ** {f"{name}__{param}": val for param, val in params.items()}
            }
        return fitted_params


class EstimatorInspector():
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
        return self.estimator
