"""Provide base classes for pipeline and estimator inspectors."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL

import re

from sklearn.utils.validation import check_is_fitted

from ..transformers import JuColumnTransformer


class PipelineInspector:
    """Provide inspector for pipeline.

    Parameters
    ----------
    model : Pipeline
        The pipeline to inspect.

    """

    def __init__(self, model):
        check_is_fitted(model)
        self._model = model

    def get_step_names(self):
        """Get the names of the steps in the pipeline.

        Returns
        -------
        list
            The names of the steps in the pipeline.

        """
        model = self._model
        if hasattr(model, "best_estimator_"):
            model = model.best_estimator_
        return list(model.named_steps.keys())

    def get_step(self, name, as_estimator=False):
        """Get a step from the pipeline.

        Parameters
        ----------
        name : str
            The name of the step to retrieve.
        as_estimator : bool, optional
            Whether to return the step as an estimator inspector or not.

        Returns
        -------
        Union[Pipeline, _EstimatorInspector]
            The requested step.

        """
        model = self._model
        if hasattr(model, "best_estimator_"):
            model = model.best_estimator_
        step = model.named_steps[name]
        if not as_estimator:
            step = _EstimatorInspector(step)
        return step

    def get_params(self):
        """Get the parameters of the pipeline.

        Returns
        -------
        dict
            The parameters of the pipeline.

        """

        return self._model.get_params()

    def get_fitted_params(self):
        """Get the fitted parameters of the pipeline.

        Get the fitted parameters of the pipeline. This includes both
        hyperparameters and fitted parameters.

        Returns
        -------
        dict
            The fitted parameters of the pipeline.

        """
        fitted_params = {}
        model = self._model
        if hasattr(model, "best_estimator_"):
            model = model.best_estimator_
        for name, step in model.steps:
            params = _EstimatorInspector(step).get_fitted_params()
            fitted_params = {
                **fitted_params,
                **{f"{name}__{param}": val for param, val in params.items()},
            }
        return fitted_params


class _EstimatorInspector:
    def __init__(self, estimator):
        self._estimator = estimator

    def get_params(self):
        return self._estimator.get_params()

    def get_fitted_params(self):
        all_params = vars(self._estimator)
        if isinstance(self._estimator, JuColumnTransformer):
            all_params = {
                **all_params,
                **vars(
                    self._estimator.column_transformer_.transformers_[0][1]
                ),
            }

        private_params = {
            param: val
            for param, val in all_params.items()
            if re.match(r"^[a-zA-Z].*[a-zA-Z0-9]*_$", param)
        }
        out = self.get_params()
        out.update(private_params)
        return out

    @property
    def estimator(self):
        return self._estimator
