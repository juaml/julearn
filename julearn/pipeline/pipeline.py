import numpy as np
from sklearn.compose import (
    ColumnTransformer, make_column_selector,
    TransformedTargetRegressor)
from sklearn.pipeline import Pipeline
from dataclasses import dataclass
from typing import Any

from .. transformers import (
    get_transformer, list_transformers,
    SetColumnTypes, JuTransformer
)
from .. estimators import list_models, get_model
from .. utils import raise_error, warn
from .. prepare import prepare_model_params


class NoInversePipeline(Pipeline):
    def inverse_transform(self, X):
        return X


def make_type_selector(pattern):

    def get_renamer(X):
        return {x: (x
                if "__:type:__" in x
                else f"{x}__:type:__continuous"
                    )
                for x in X.columns
                }

    def type_selector(X):

        renamer = get_renamer(X)
        _X = X.rename(columns=renamer)
        reverse_renamer = {
            new_name: name
            for name, new_name in renamer.items()}
        selected_columns = make_column_selector(pattern)(_X)
        return [reverse_renamer[col] if col in reverse_renamer else col
                for col in selected_columns]

    return type_selector


def get_default_patter(step):
    return "__:type:__continuous"


def get_step(step_name, pattern):
    return step_name, get_transformer(step_name), pattern


@dataclass
class Step:
    name: str
    estimator: Any
    apply_to: Any = "continuous"
    params: dict = None

    def __post_init__(self):
        self.params = {} if self.params is None else self.params


class PipelineCreator:  # Pipeline creator
    def __init__(self):
        self._steps = list()
        self._added_target_transformer = False
        self._added_model = False

    def add(self, step, apply_to="continuous",
            problem_type="binary_classification", **params):
        apply_to = self._ensure_apply_to(apply_to)
        self.validate_step(step, apply_to)
        name = step if isinstance(step, str) else step.__cls__.lower()
        name = self._ensure_name(name)
        estimator = (
            self.get_estimator_from(step, problem_type)
            if isinstance(step, str)
            else step
        )
        params_to_set = dict()
        for param, vals in params.items():
            if len(vals) == 1:
                params_to_set[param] = params.pop(param)

        estimator = estimator.set_params(**params_to_set)
        if apply_to == "target":
            name = f"target_{name}"

        self._steps.append(
            Step(name=name,
                 estimator=estimator,
                 apply_to=apply_to,
                 params=params)
        )
        return self

    @property
    def steps(self):
        return self._steps

    @classmethod
    def from_list(cls, transformers: list):
        preprocessor = cls()
        for transformer_name in transformers:
            preprocessor.add(transformer_name)
        return preprocessor

    def to_pipeline(self, X_types, model_params=None,
                    model=None, problem_type="binary_classification"):

        pipeline_steps = [("set_column_types", SetColumnTypes(X_types))]
        if model_params is None:
            model_params = {}
            search_params = None
        else:
            search_params = (model_params.pop("search_params")
                             if "search_params" in model_params
                             else None)

        if self._added_model and model is not None:
            raise_error()
        self._validate_model_params(model_params)

        self.ensure_X_types(X_types)
        wrap = list(X_types.keys()) != ["continuous"]

        transformer_steps = self._steps
        target_transformer_steps = []
        if self._added_model:
            transformer_steps = self._steps[:-1]
            model_step = self._steps[-1]
        if self._added_target_transformer:
            _transformer_steps = []
            for _step in self._steps:
                if _step.apply_to == "target":
                    target_transformer_steps.append(_step)
                else:
                    _transformer_steps.append(_step)
            transformer_steps = _transformer_steps

        step_params = {}
        for step_dict in transformer_steps:
            name = step_dict.name
            est_param_name = name
            estimator = step_dict.estimator

            transformer_model_params = {
                param: model_params.pop(param)
                for param, _ in model_params.items()
                if param.startswith(name + '__')
            }

            if wrap and not isinstance(estimator, JuTransformer):
                # TODO check is julearn esti
                estimator = self.wrap_step(
                    name, estimator, step_dict.apply_to)
                est_param_name = f"wrappend_{name}__{name}"
                name = f"wrapped_{name}"

            pipeline_steps.append((name, estimator))

            transformer_params = {
                **transformer_model_params, **step_dict.params}
            step_params = {**step_params, **{
                f"{est_param_name}__{param}": val
                for param, val in transformer_params.items()}
            }

        if model is not None:
            model_est = (get_model(model, problem_type)
                         if isinstance(model, str)
                         else model)
            model_name = (model
                          if isinstance(model, str)
                          else model.__cls__.lower
                          )
            model_step = Step(
                name=model_name, estimator=model_est,
                params={}, apply_to=None
            )

        model_step, model_params = self.wrap_target_model(
            model_step, target_transformer_steps,
            model_params
        )
        pipeline_steps.append(model_step)

        pipeline = Pipeline(pipeline_steps).set_output(transform="pandas")
        # Deal with the CV
        model_params["search_params"] = search_params
        return prepare_model_params(model_params, pipeline)

    @staticmethod
    def wrap_target_model(
            model_step, target_transformer_steps, model_params=None):
        model_params = {} if model_params is None else model_params
        # TODO adjust params
        if target_transformer_steps == []:
            return (model_step.name, model_step.estimator), model_params
        transformer_pipe = NoInversePipeline(target_transformer_steps)
        target_model = TransformedTargetRegressor(
            transformer=transformer_pipe,
            regressor=model_step[1],
            check_inverse=False
        )
        return (f"{model_step[0]}_target_transform",
                target_model)

    def _validate_model_params(self, model_params):

        for param in model_params.keys():
            if "__" in param:
                est_name = param.split("__")[0]
                if est_name in [step_dict.name
                                for step_dict in self._steps]:
                    raise_error("")  # TODO

    def _ensure_name(self, name):

        count = np.array(
            [_step.name == name
             for _step in self._steps
             ]).sum()
        return f"{name}_{count}" if count > 0 else name

    def validate_step(self, step, apply_to):
        if self._is_transfromer_step(step):
            if self._added_model:
                raise_error()  # TODO
            if self._added_target_transformer and apply_to != "target":
                raise_error()  # TODO
            if apply_to == "target":
                self._added_target_transformer = True
        elif self._is_model_step(step):
            self._added_model = True
        else:
            raise_error()  # TODO

    def ensure_X_types(self, X_types):
        all_types = list(X_types.keys())
        unique_apply_to = []
        for step_dict in self._steps:
            _apply_to = step_dict.apply_to
            if isinstance(_apply_to, str):
                unique_apply_to.append(_apply_to.split("__")[-1])
            else:
                for _apply in _apply_to:
                    unique_apply_to.append(_apply.split("__")[-1])
        unique_apply_to = set(unique_apply_to)
        for X_type in all_types:
            if X_type not in unique_apply_to:
                warn(
                    f"{X_type} is provided but never used by a transformer."
                    f"Used types are {unique_apply_to}"
                )

        # TODO:
        # put logic here
        # On JulearnTransformers: Raise Error if there is
        # a needed type that is not in the X_type

        return [f"__:type:__{X_type}" for X_type in all_types]

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
    def wrap_step(name, step, pattern):
        return ColumnTransformer(
            [(name, step, make_type_selector(pattern))],
            verbose_feature_names_out=False, remainder="passthrough"
        )

    @staticmethod
    def _ensure_apply_to(apply_to):
        if isinstance(apply_to, list) or isinstance(apply_to, tuple):
            types = [f"__:type:__{_type}" for _type in apply_to]

            pattern = f"({types[0]}"
            if len(types) > 1:
                for t in types[1:]:
                    pattern += fr"|{t}"
            pattern += r")"
        else:
            pattern = f"__:type:__{apply_to}"
        return pattern

    @staticmethod
    def get_estimator_from(name, problem_type):
        if name in list_transformers():
            return get_transformer(name)
        if name in list_models():
            return get_model(name, problem_type)
        raise_error(
            f"{name} is neither a registered transformer"
            "nor a registered model."
        )
