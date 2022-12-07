from typing import Any, Union, List, Dict, Optional, Tuple

import numpy as np
from sklearn.compose import (
    TransformedTargetRegressor,
)
from sklearn.pipeline import Pipeline
from sklearn.model_selection import check_cv
from dataclasses import dataclass, field

from ..transformers import (
    get_transformer,
    list_transformers,
    SetColumnTypes,
)
from ..models import list_models, get_model
from ..utils import raise_error, warn, logger
from ..base import ColumnTypes, WrapModel, JuTransformer
from ..utils.typing import JuModelLike, JuEstimatorLike
from ..transformers import JuColumnTransformer
from ..model_selection.available_searchers import list_searchers, get_searcher


class NoInversePipeline(Pipeline):
    def inverse_transform(self, X):
        return X


def _params_to_pipeline(param, X_types):
    if isinstance(param, PipelineCreator):
        param = param.to_pipeline(X_types=X_types)
    elif isinstance(param, list):
        param = [_params_to_pipeline(_v, X_types) for _v in param]
    elif isinstance(param, dict):
        param = {
            k: _params_to_pipeline(_v, X_types) for k, _v in param.items()
        }
    elif isinstance(param, tuple):
        param = tuple(_params_to_pipeline(_v, X_types) for _v in param)
    return param


@dataclass
class Step:
    name: str
    estimator: Any
    apply_to: ColumnTypes = field(
        default_factory=lambda: ColumnTypes("continuous")
    )
    needed_types: Any = None
    params_to_tune: Optional[Dict] = None

    def __post_init__(self):
        self.params_to_tune = (
            {} if self.params_to_tune is None else self.params_to_tune
        )


class PipelineCreator:  # Pipeline creator
    """PipelineCreator class.

    Parameters
    ----------
    problem_type: {"classification", "regression"}
        The problem type for which this pipeline should be created.
    """

    def __init__(self, problem_type, apply_to="continuous"):
        if problem_type not in ["classification", "regression"]:
            raise_error(
                "`problem_type` should be either 'classification' or "
                "'regression'."
            )
        self._steps = list()
        self._added_target_transformer = False
        self._added_model = False
        self.apply_to = apply_to
        self.problem_type = problem_type

    def add(
        self,
        step,
        name=None,
        apply_to=None,
        **params,
    ):
        """Add a step to the PipelineCreator.
        This includes transformers and models.

        Parameters
        ----------
        step : str or TransformerLike or ModelLike
            The step that should be added.
            This can be an available_transformer or
            available_model as a str or a sklearn compatible
            transformer or model.
        apply_to: str or list of str or ColumnTypes, Optional
            To what should the transformer or model be applied to.
            This can be a str representing a column type or a list
            of such str.
        **params
            Parameters for the step. This will mostly include hyperparameters
            or any other parameter for initialization.
            If you provide multiple options for hyperparameters then
            this will lead to a pipeline with a search which is by
            default GridSearchCV.

        Returns
        -------
        PipelineCreator: PipelineCreator
        returns a PipelineCreator with the added step as its last step.
        """

        apply_to = self.apply_to if apply_to is None else apply_to
        apply_to = ColumnTypes(apply_to)
        self.validate_step(step, apply_to)
        if name is None:
            name = step if isinstance(step, str) else step.__cls__.lower()
            name = self._ensure_name(name)
        logger.info(f"Adding step {name} that applies to {apply_to}")
        params_to_set = dict()
        params_to_tune = dict()
        for param, vals in params.items():
            # If we have more than 1 value, we will tune it.
            # If not, it will be set in the model.
            if hasattr(vals, "__iter__") and not isinstance(vals, str):
                if len(vals) > 1:
                    logger.info(f"Tuning hyperparameter {param} = {vals}")
                    params_to_tune[param] = vals
                else:
                    logger.info(f"Setting hyperparameter {param} = {vals[0]}")
                    params_to_set[param] = vals[0]
            else:
                logger.info(f"Setting hyperparameter {param} = {vals}")
                params_to_set[param] = vals
        problem_type = (params_to_set.pop("problem_type")
                        if "problem_type" in params_to_set
                        else self.problem_type)
        estimator = (
            self.get_estimator_from(step, problem_type, **params_to_set)
            if isinstance(step, str)
            else step
        )
        if isinstance(estimator, JuEstimatorLike):
            estimator.set_params(apply_to=apply_to)
            needed_types = estimator.get_needed_types()
        else:
            needed_types = apply_to
        if apply_to == "target":
            name = f"target_{name}"
        self._steps.append(
            Step(
                name=name,
                estimator=estimator,
                apply_to=apply_to,
                needed_types=needed_types,
                params_to_tune=params_to_tune,
            )
        )
        logger.info("Step added")
        return self

    @property
    def steps(self):
        return self._steps

    def has_model(self) -> bool:
        return self._added_model

    @classmethod
    def from_list(
        cls,
        transformers: Union[str, list],
        model_params: dict,
        problem_type: str,
    ):
        preprocessor = cls(problem_type)
        if isinstance(transformers, str):
            transformers = [transformers]
        for transformer_name in transformers:
            t_params = {
                x.replace(f"{transformer_name}__", ""): y
                for x, y in model_params.items()
                if x.startswith(f"{transformer_name}__")
            }
            preprocessor.add(transformer_name, **t_params)
        return preprocessor

    def to_pipeline(
        self, X_types: Optional[Dict[str, List]] = None, search_params=None
    ):

        logger.debug("Creating pipeline")
        if not self.has_model():
            raise_error("Cannot create a pipeline without a model")
        pipeline_steps: List[Tuple[str, Any]] = [
            ("set_column_types", SetColumnTypes(X_types))
        ]

        self.check_X_types(X_types)
        model_step = self._steps[-1]

        target_transformer_steps = []
        transformer_steps = []

        for _step in self._steps[:-1]:
            if _step.apply_to == "target":
                target_transformer_steps.append(_step)
            else:
                transformer_steps.append(_step)

        # Add transformers
        params_to_tune = {}
        for step_dict in transformer_steps:
            logger.debug(f"Adding transformer {step_dict.name}")
            name = step_dict.name
            estimator = step_dict.estimator
            logger.debug(f"\t Estimator: {estimator}")
            step_params_to_tune = step_dict.params_to_tune
            logger.debug(f"\t Params to tune: {step_params_to_tune}")

            # Wrap in a JuTransformer if needed
            if self.wrap and not isinstance(estimator, JuTransformer):
                estimator = self.wrap_step(name, estimator, step_dict.apply_to)

            pipeline_steps.append((name, estimator))

            # Add params to tune
            params_to_tune.update(
                {
                    f"{name}__{param}": val
                    for param, val in step_params_to_tune.items()
                }
            )

        model_name = model_step.name
        model_estimator = model_step.estimator
        logger.debug(f"Adding model {model_name}")

        model_params = model_estimator.get_params(deep=False)
        model_params = {
            k: _params_to_pipeline(v, X_types=X_types)
            for k, v in model_params.items()
        }
        model_estimator.set_params(**model_params)
        if self.wrap and not isinstance(model_estimator, JuModelLike):
            logger.debug(f"Wrapping {model_name}")
            model_estimator = WrapModel(model_estimator, model_step.apply_to)

        step_params_to_tune = {
            f"{model_name}__{k}": v
            for k, v in model_step.params_to_tune.items()
        }

        logger.debug(f"\t Estimator: {model_estimator}")
        logger.debug("\t Looking for nested pipeline creators")
        logger.debug(f"\t Params to tune: {step_params_to_tune}")
        if self._added_target_transformer:
            target_model_step, step_params_to_tune = self.wrap_target_model(
                model_name,
                model_estimator,
                target_transformer_steps,
                step_params_to_tune,
            )
            params_to_tune.update(step_params_to_tune)
            pipeline_steps.append(target_model_step)
        else:

            params_to_tune.update(step_params_to_tune)
            pipeline_steps.append((model_name, model_estimator))
        pipeline = Pipeline(pipeline_steps).set_output(transform="pandas")

        # Deal with the Hyperparameter tuning
        out = self.prepare_hyperparameter_tuning(
            params_to_tune, search_params, pipeline
        )
        logger.debug("Pipeline created")
        return out

    @staticmethod
    def prepare_hyperparameter_tuning(params_to_tune, search_params, pipeline):
        """Prepare model parameters.

        For each of the model parameters, determine if it can be directly set
        or must be tuned using hyperparameter tuning.

        Parameters
        ----------
        msel_dict : dict
            A dictionary with the model selection parameters.The dictionary can
            define the following keys:

            * 'STEP__PARAMETER': A value (or several) to be used as PARAMETER
            for STEP in the pipeline. Example: 'svm__probability':
            True will set
            the parameter 'probability' of the 'svm' model. If more than option
            * 'search': The kind of search algorithm to use e.g.:
            'grid' or 'random'. All valid julearn searchers can be entered.
            * 'cv': If search is going to be used, the cross-validation
            splitting strategy to use. Defaults to same CV as for the model
            evaluation.
            * 'scoring': If search is going to be used, the scoring metric to
            evaluate the performance.
            * 'search_params': Additional parameters for the search method.

        pipeline : ExtendedDataframePipeline
            The pipeline to apply/tune the hyperparameters

        Returns
        -------
        pipeline : ExtendedDataframePipeline
            The modified pipeline
        """
        logger.info("= Model Parameters =")

        search_params = {} if search_params is None else search_params
        if len(params_to_tune) > 0:
            search = search_params.get("kind", "grid")
            scoring = search_params.get("scoring", None)
            cv_inner = search_params.get("cv", None)

            if search in list_searchers():
                logger.info(f"Tuning hyperparameters using {search}")
                search = get_searcher(search)
            else:
                if isinstance(search, str):
                    raise_error(
                        f"The searcher {search} is not a valid julearn"
                        " searcher. "
                        "You can get a list of all available once by using: "
                        "julearn.model_selection.list_searchers(). "
                        "You can also "
                        "enter a valid scikit-learn searcher or register it."
                    )
                else:
                    warn(f"{search} is not a registered searcher. ")
                    logger.info(
                        f"Tuning hyperparameters using not registered {search}"
                    )

            logger.info("Hyperparameters:")
            for k, v in params_to_tune.items():
                logger.info(f"\t{k}: {v}")

            cv_inner = check_cv(cv_inner)  # type: ignore
            logger.info(f"Using inner CV scheme {cv_inner}")
            search_params["cv"] = cv_inner
            search_params["scoring"] = scoring
            logger.info("Search Parameters:")
            for k, v in search_params.items():
                logger.info(f"\t{k}: {v}")
            pipeline = search(pipeline, params_to_tune, **search_params)
        elif search_params is not None and len(search_params) > 0:
            warn(
                "Hyperparameter search parameters were specified, but no "
                "hyperparameters to tune"
            )
        logger.info("====================")
        logger.info("")
        return pipeline

    @staticmethod
    def wrap_target_model(
        model_name, model, target_transformer_steps, model_params=None
    ):
        model_params = {} if model_params is None else model_params
        model_params = {
            f"regressor__{param}": val for param, val in model_params.items()
        }

        def check_has_valid_reverse(est):
            valid_reverse = True
            if isinstance(est, Pipeline):
                for _step in est.steps:
                    if not check_has_valid_reverse(_step[1]):
                        return False
            else:
                if not hasattr(est, "inverse_transform"):
                    valid_reverse = False
                return valid_reverse

        pipe_trans_steps = []
        valid_reverse = True
        for step in target_transformer_steps:
            name, est = step.name, step.estimator
            pipe_trans_steps.append([name, est])
            if not check_has_valid_reverse(est):
                valid_reverse = False

        transformer_pipe = (
            Pipeline(pipe_trans_steps)
            if valid_reverse
            else NoInversePipeline(pipe_trans_steps)
        )
        target_model = TransformedTargetRegressor(
            transformer=transformer_pipe.set_output(transform="pandas"),
            regressor=model,
            check_inverse=False,
        )
        return (f"{model_name}_target_transform", target_model), model_params

    def _validate_model_params(self, model_name, model_params):

        for param in model_params.keys():
            if "__" in param:
                est_name = param.split("__")[0]
                if est_name != model_name:
                    raise_error(
                        "Only parameters for the model should be specified. "
                        f"Got {param} for {est_name}."
                    )

    def _ensure_name(self, name):

        count = np.array([_step.name == name for _step in self._steps]).sum()
        return f"{name}_{count}" if count > 0 else name

    def validate_step(self, step, apply_to):

        if self._is_transfromer_step(step):
            if self._added_model:
                raise_error("Cannot add a transformer after adding a model")
            if self._added_target_transformer and apply_to != "target":
                raise_error(
                    "Cannot add a non-target transformer after adding "
                    "a target transformer."
                )
            if apply_to == "target":
                self._added_target_transformer = True
        elif self._is_model_step(step):
            self._added_model = True
        else:
            raise_error(f"Cannot add a {step}. I don't know what it is.")

    def check_X_types(self, X_types: Optional[Dict] = None):
        # Get the set of all types in the X_types
        if X_types is None or X_types == {}:
            all_X_types = ColumnTypes("continuous")
        else:
            all_X_types = ColumnTypes(list(X_types.keys()))

        # Get the set of all needed types by the pipeline
        needed_types = []
        for step_dict in self._steps:
            if step_dict.needed_types is None:
                continue
            needed_types.extend(step_dict.needed_types)
        needed_types = set(needed_types)

        skip_need_error = ".*" in needed_types or "*" in needed_types
        # applied_to_special = (needed_types == "continuous" or
        #                       needed_types == "target" or
        #                       ".*" in needed_types or
        #                       "*" in needed_types)
        if not skip_need_error:
            extra_types = [x for x in all_X_types if x not in needed_types]
            if len(extra_types) > 0:
                raise_error(
                    f"Extra X_types were provided but never used by a "
                    f"transformer.\n"
                    f"\tExtra types are {extra_types}\n"
                    f"\tUsed types are {needed_types}"
                )

        # All available types are the ones in the X_types + wildcard types +
        # target + the ones that can be created by a transformer.
        # So far, we only know of transformers that output continuous
        available_types = set(
            [*all_X_types, "*", ".*", "target", "continuous"]
        )
        for needed_type in needed_types:
            if needed_type not in available_types:
                warn(
                    f"{needed_type} is not in the provided X_types={X_types}. "
                    "Make sure your pipeline has a transformer that creates "
                    "this type."
                )

        self.wrap = needed_types != set(["continuous"])

    @staticmethod
    def _is_transfromer_step(step):
        if step in list_transformers():
            return True
        if hasattr(step, "fit") and hasattr(step, "transform"):
            return True
        return False

    @staticmethod
    def _is_model_step(step):
        if step in list_models():
            return True
        if hasattr(step, "fit") and hasattr(step, "predict"):
            return True
        return False

    @staticmethod
    def wrap_step(name, step, column_types):
        return JuColumnTransformer(name, step, column_types)

    @staticmethod
    def get_estimator_from(name, problem_type, **kwargs):
        if name in list_transformers():
            return get_transformer(name, **kwargs)
        if name in list_models():
            return get_model(name, problem_type, **kwargs)
        raise_error(
            f"{name} is neither a registered transformer"
            "nor a registered model."
        )
