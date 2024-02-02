"""PipelineCreator class."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.model_selection import check_cv
from sklearn.pipeline import Pipeline

from ..base import ColumnTypes, ColumnTypesLike, JuTransformer, WrapModel
from ..model_selection.available_searchers import get_searcher, list_searchers
from ..models import get_model, list_models
from ..prepare import prepare_search_params
from ..transformers import (
    JuColumnTransformer,
    SetColumnTypes,
    get_transformer,
    list_transformers,
)
from ..transformers.target import JuTransformedTargetModel
from ..utils import logger, raise_error, warn_with_log
from ..utils.typing import (
    EstimatorLike,
    JuEstimatorLike,
    JuModelLike,
    ModelLike,
)
from .target_pipeline import JuTargetPipeline
from .target_pipeline_creator import TargetPipelineCreator


def _params_to_pipeline(
    param: Any,
    X_types: Dict[str, List],  # noqa: N803
    search_params: Optional[Dict],
):
    """Recursively convert params to pipelines.

    Parameters
    ----------
    param : Any
        The parameter to convert.
    X_types : Dict[str, List]
        The types of the columns in the data.
    search_params : Optional[Dict]
        The parameters to tune for this step, by default None

    Returns
    -------
    Any
        The converted parameter.

    """
    if isinstance(param, PipelineCreator):
        param = param.to_pipeline(X_types=X_types, search_params=search_params)
    elif isinstance(param, list):
        param = [
            _params_to_pipeline(_v, X_types, search_params) for _v in param
        ]
    elif isinstance(param, dict):
        param = {
            k: _params_to_pipeline(_v, X_types, search_params)
            for k, _v in param.items()
        }
    elif isinstance(param, tuple):
        param = tuple(
            _params_to_pipeline(_v, X_types, search_params) for _v in param
        )
    return param


@dataclass
class Step:
    """Step class.

    This class represents a step in a pipeline.


    Parameters
    ----------
    name : str
        The name of the step.
    estimator : Any
        The estimator to use.
    apply_to : ColumnTypesLike
        The types to apply this step to, by default "continuous"
    needed_types : Any, optional
        The types needed by this step (default is None)
    row_select_col_type : str or list of str or set of str or ColumnTypes
        The column types needed to select rows (default is None)
    row_select_vals : str, int, bool or list of str, int, bool
        The value(s) which should be selected in the row_select_col_type
        to select the rows used for training (default is None)
    params_to_tune : Optional[Dict], optional
        The parameters to tune for this step, by default None

    """

    name: str
    estimator: Union[JuEstimatorLike, EstimatorLike]
    apply_to: ColumnTypes = field(
        default_factory=lambda: ColumnTypes("continuous")
    )
    needed_types: Optional[ColumnTypesLike] = None
    params_to_tune: Optional[Dict] = None

    row_select_col_type: Optional[ColumnTypesLike] = None
    row_select_vals: Optional[Union[str, int, list, bool]] = None

    def __post_init__(self) -> None:
        """Post init."""
        self.params_to_tune = (
            {} if self.params_to_tune is None else self.params_to_tune
        )


class PipelineCreator:
    """PipelineCreator class.

    This class is used to create pipelines. As the creation of a pipeline
    is a bit more complicated than just adding steps to a pipeline, this
    helper class is provided so the user can easily create complex
    :class:`sklearn.pipeline.Pipeline` objects.

    Parameters
    ----------
    problem_type: {"classification", "regression"}
        The problem type for which the pipeline should be created.
    apply_to: ColumnTypesLike, optional
        To what should the transformers be applied to if not specified in
        the `add` method (default is continuous).

    """

    def __init__(
        self, problem_type: str, apply_to: ColumnTypesLike = "continuous"
    ):
        if problem_type not in ["classification", "regression"]:
            raise_error(
                "`problem_type` should be either 'classification' or "
                "'regression'."
            )
        self._steps = []
        self._added_target_transformer = False
        self._added_model = False
        self.apply_to = apply_to
        self.problem_type = problem_type

    def add(
        self,
        step: Union[EstimatorLike, str, TargetPipelineCreator],
        name: Optional[str] = None,
        apply_to: Optional[ColumnTypesLike] = None,
        row_select_col_type: Optional[ColumnTypesLike] = None,
        row_select_vals: Optional[Union[str, int, list, bool]] = None,
        **params: Any,
    ) -> "PipelineCreator":
        """Add a step to the PipelineCreator.

        Parameters
        ----------
        step : EstimatorLike
            The step that should be added.
            This can be an available_transformer or
            available_model as a str or a sklearn compatible
            transformer or model.
        name : str, optional
            The name of the step. If None, the name will be obtained from
            the step (default is None).
        apply_to: ColumnTypesLike, optional
            To what should the transformer or model be applied to.
            This can be a str representing a column type or a list
            of such str (defaults to the `PipelineCreator.apply_to` attribute).
        row_select_col_type : str or list of str or set of str or ColumnTypes
            The column types needed to select rows (default is None)
        row_select_vals : str, int, bool or list of str, int, bool
            The value(s) which should be selected in the row_select_col_type
            to select the rows used for training (default is None)
        **params
            Parameters for the step. This will mostly include
            hyperparameters or any other parameter for initialization.
            If you provide multiple options for hyperparameters then
            this will lead to a pipeline with a search.

        Returns
        -------
        PipelineCreator
            The PipelineCreator with the added step as its last step.

        Raises
        ------
        ValueError
            If the step is not a valid step, if the problem_type is
            specified in the params or if the step is a
            TargetPipelineCreator and the apply_to is not "target".

        """

        if "problem_type" in params:
            raise_error(
                "Please provide the problem_type directly"
                " and only to the PipelineCreator like this"
                " PipelineCreator(problem_type=problem_type)"
            )
        apply_to = self.apply_to if apply_to is None else apply_to
        apply_to = ColumnTypes(apply_to)

        if isinstance(step, TargetPipelineCreator):
            if apply_to != "target":
                raise_error(
                    "TargetPipelineCreator can only be added to the target."
                )
            step = step.to_pipeline()  # type: ignore

        # Validate the step
        self._validate_step(step, apply_to)

        # Check that if the name is already in the steps, we are adding them
        # at the same place.
        if name is not None and name in [step.name for step in self._steps]:
            if self._steps[-1].name != name:
                raise_error(
                    "Repeated step names are only allowed to be added "
                    "consecutively. That means that all the steps with the "
                    "same name should be added one after the other."
                    f"The step {name} was already added."
                )

        # If the user did not give a name, we will create one.
        name = self._get_step_name(name, step)
        logger.info(f"Adding step {name} that applies to {apply_to}")

        # Find which parameters should be set and which should be tuned.
        params_to_set = {}
        params_to_tune = {}
        for param, vals in params.items():
            # If we have more than 1 value, we will tune it.
            # If not, it will be set in the model.
            if (
                hasattr(vals, "__iter__")
                and not hasattr(vals, "fit")
                and not isinstance(vals, str)
            ):
                if len(vals) > 1:
                    logger.info(f"Tuning hyperparameter {param} = {vals}")
                    params_to_tune[param] = vals
                else:
                    logger.info(f"Setting hyperparameter {param} = {vals[0]}")
                    params_to_set[param] = vals[0]
            else:
                logger.info(f"Setting hyperparameter {param} = {vals}")
                params_to_set[param] = vals

        # Build the estimator for this step
        if isinstance(step, str):
            step = self._get_estimator_from(
                step, self.problem_type, **params_to_set
            )
        elif len(params_to_set) > 0:
            step.set_params(**params_to_set)

        # JuEstimators accept the apply_to parameter and return needed types
        if isinstance(step, JuEstimatorLike):
            # But some JuEstimators might fix the apply_to parameter
            if "apply_to" in step.get_params(deep=False):
                step.set_params(apply_to=apply_to)
            needed_types = step.get_needed_types()
        elif isinstance(step, JuTargetPipeline):
            needed_types = apply_to
            if step.needed_types is not None:
                needed_types.add(step.needed_types)
        else:
            needed_types = apply_to

        if isinstance(step, JuTransformer):
            step.set_params(
                row_select_col_type=row_select_col_type,
                row_select_vals=row_select_vals,
            )

            needed_types = step.get_needed_types()

        # For target transformers we need to add the target_ prefix
        if apply_to == "target":
            name = f"target_{name}"

        # Fix the params to tune keys:
        params_to_tune = {
            f"{name}__{param}": val for param, val in params_to_tune.items()
        }

        self._steps.append(
            Step(
                name=name,
                estimator=step,
                apply_to=apply_to,
                needed_types=needed_types,
                params_to_tune=params_to_tune,
                row_select_col_type=row_select_col_type,
                row_select_vals=row_select_vals,
            )
        )
        logger.info("Step added")
        return self

    @property
    def steps(self) -> List[Step]:
        """Get the steps that have been added to the PipelineCreator."""
        return self._steps

    def __str__(self) -> str:
        """Get a string representation of the PipelineCreator."""
        out = "PipelineCreator:\n"
        for i_step, step in enumerate(self._steps):
            out += f"  Step {i_step}: {step.name}\n"
            out += f"    estimator:     {step.estimator}\n"
            out += f"    apply to:      {step.apply_to}\n"
            out += f"    needed types:  {step.needed_types}\n"
            out += f"    tuning params: {step.params_to_tune}\n"
        return out

    def has_model(self) -> bool:
        """Whether the PipelineCreator has a model."""
        return self._added_model

    def copy(self) -> "PipelineCreator":
        """Create a copy of the PipelineCreator.

        Returns
        -------
        PipelineCreator
            The copy of the PipelineCreator

        """
        other = PipelineCreator(
            problem_type=self.problem_type, apply_to=self.apply_to
        )
        other._steps = self._steps.copy()
        return other

    @classmethod
    def from_list(
        cls,
        transformers: Union[str, list],
        model_params: dict,
        problem_type: str,
        apply_to: ColumnTypesLike = "continuous",
    ) -> "PipelineCreator":
        """Create a PipelineCreator from a list of transformers and parameters.

        Parameters
        ----------
        transformers : Union[str, list]
            The transformers that should be added to the PipelineCreator.
            This can be a str or a list of str.
        model_params : dict
            The parameters for the model and the transformers.
            This should be a dict with the keys being the name of the
            transformer or the model and the values being a dict with
            the parameters for that transformer or model.
        problem_type : str
            The problem_type for which the piepline should be created.
        apply_to : ColumnTypesLike, optional
            To what should the transformers be applied to if not specified in
            the `add` method (default is continuous).

        Returns
        -------
        PipelineCreator
            The PipelineCreator with the steps added

        """
        creator = cls(problem_type=problem_type, apply_to=apply_to)
        if isinstance(transformers, str):
            transformers = [transformers]
        for transformer_name in transformers:
            t_params = {
                x.replace(f"{transformer_name}__", ""): y
                for x, y in model_params.items()
                if x.startswith(f"{transformer_name}__")
            }
            creator.add(transformer_name, **t_params)
        return creator

    def split(
        self,
    ) -> List["PipelineCreator"]:
        """Split the PipelineCreator into multiple PipelineCreators.

        If the PipelineCreator has at least two steps with the same name,
        this is considered a split point for hyperparameter tuning.
        This function will split the PipelineCreator into multiple
        PipelineCreators, one for each split point, recursively. Thus, creating
        as many PipelineCreators as needed to tune all the hyperparameters
        configurations.


        Returns
        -------
        List[PipelineCreator]
            A list of PipelineCreators, each one without repeated step names.

        """

        out = []
        # Add the default PipelineCreator with the same parameters.
        out.append(
            PipelineCreator(
                problem_type=self.problem_type, apply_to=self.apply_to
            )
        )
        names = [x.name for x in self._steps]

        # Create a list of unique names, keeping the order
        unique_names = []
        for name in names:
            if name not in unique_names:
                unique_names.append(name)

        for t_name in unique_names:
            # Get the list of steps that should be added to the PipelineCreator
            t_steps = [x for x in self._steps if x.name == t_name]

            # Check how many steps we have with that name
            if len(t_steps) == 1:
                # If it is only one, add it to each of the PipelineCreators
                for t_pipe in out:
                    t_pipe._steps.append(t_steps[0])
            else:
                # If we have more than one, we need to create a new
                # PipelineCreator for each of the steps.
                new_out = []
                for t_pipe in out:
                    for t_step in t_steps:
                        new_pipe = t_pipe.copy()
                        new_pipe._steps.append(t_step)
                        new_out.append(new_pipe)
                out = new_out
        for t_out in out:
            t_out._added_model = self._added_model
            t_out._added_target_transformer = self._added_target_transformer
        return out

    def to_pipeline(
        self,
        X_types: Optional[Dict[str, List]] = None,  # noqa: N803
        search_params: Optional[Dict[str, Any]] = None,
    ) -> Pipeline:
        """Create a pipeline from the PipelineCreator.

        Parameters
        ----------
        X_types : Optional[Dict[str, List]], optional
            The types of the columns in the data, by default None
        search_params : Optional[Dict], optional
            The parameters for the search, by default None

        Returns
        -------
        sklearn.pipeline.Pipeline
            The pipeline created from the PipelineCreator

        """
        logger.debug("Creating pipeline")
        if not self.has_model():
            raise_error("Cannot create a pipeline without a model")
        pipeline_steps: List[Tuple[str, Any]] = [
            ("set_column_types", SetColumnTypes(X_types))
        ]

        X_types = self._check_X_types(X_types)
        model_step = self._steps[-1]

        target_transformer_step = None
        transformer_steps = []

        for _step in self._steps[:-1]:
            if "target" in _step.apply_to:
                target_transformer_step = _step
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
                estimator = self._wrap_step(
                    name,
                    estimator,
                    step_dict.apply_to,
                    row_select_col_type=step_dict.row_select_col_type,
                    row_select_vals=step_dict.row_select_vals,
                )

            # Check if a step with the same name was already added
            pipeline_steps.append((name, estimator))

            # Add params to tune
            params_to_tune.update(step_params_to_tune)

        # Add final model
        model_name = model_step.name
        model_estimator = model_step.estimator
        logger.debug(f"Adding model {model_name}")

        model_params = model_estimator.get_params(deep=False)
        model_params = {
            k: _params_to_pipeline(
                v, X_types=X_types, search_params=search_params
            )
            for k, v in model_params.items()
        }
        model_estimator.set_params(**model_params)
        if self.wrap and not isinstance(model_estimator, JuModelLike):
            logger.debug(f"Wrapping {model_name}")
            model_estimator = WrapModel(model_estimator, model_step.apply_to)

        step_params_to_tune = model_step.params_to_tune

        logger.debug(f"\t Estimator: {model_estimator}")
        logger.debug("\t Looking for nested pipeline creators")
        logger.debug(f"\t Params to tune: {step_params_to_tune}")
        if self._added_target_transformer:
            # If we have a target transformer, we need to wrap the model
            # in a the right "Targeted" transformer.
            target_model_step = self._wrap_target_model(
                model_name,
                model_estimator,  # type: ignore
                target_transformer_step,  # type: ignore
            )
            target_step_to_tune = {
                f"{model_name}_target_transform__transformer__{param}": val
                for param, val in (
                    target_transformer_step.params_to_tune.items()
                )
            }
            step_params_to_tune = {
                f"{model_name}_target_transform__model__{param}": val
                for param, val in step_params_to_tune.items()
            }
            pipeline_steps.append(target_model_step)
            params_to_tune.update(step_params_to_tune)
            params_to_tune.update(target_step_to_tune)
        else:
            # if not, just add a model as the last step
            params_to_tune.update(step_params_to_tune)
            pipeline_steps.append((model_name, model_estimator))
        pipeline = Pipeline(pipeline_steps).set_output(transform="pandas")

        # Deal with the Hyperparameter tuning
        out = _prepare_hyperparameter_tuning(
            params_to_tune, search_params, pipeline
        )
        logger.debug("Pipeline created")
        return out

    @staticmethod
    def _wrap_target_model(
        model_name: str, model: ModelLike, target_transformer_step: Step
    ) -> Tuple[str, JuTransformedTargetModel]:
        """Wrap the model in a JuTransformedTargetModel.

        Parameters
        ----------
        model_name : str
            The name of the model
        model : ModelLike
            The model to wrap
        target_transformer_step : Step
            The step with the target transformer.

        Returns
        -------
        str :
            The name of the model.
        JuTransformedTargetModel :
            The wrapped model.

        Raises
        ------
        ValueError
            If the target transformer is not a JuTargetPipeline.

        """
        transformer = target_transformer_step.estimator
        if not isinstance(transformer, JuTargetPipeline):
            raise_error(
                "The target transformer should be a JuTargetPipeline. "
                f"Got {type(transformer)}"
            )
        target_model = JuTransformedTargetModel(
            model=model,
            transformer=transformer,
        )
        return (f"{model_name}_target_transform", target_model)

    def _validate_model_params(
        self, model_name: str, model_params: Dict[str, Any]
    ) -> None:
        """Validate the model parameters.

        Parameters
        ----------
        model_name : str
            The name of the model.
        model_params : dict
            The parameters of the model to validate.

        Raises
        ------
        ValueError
            If the model parameters are not valid.

        """
        for param in model_params.keys():
            if "__" in param:
                est_name = param.split("__")[0]
                if est_name != model_name:
                    raise_error(
                        "Only parameters for the model should be specified. "
                        f"Got {param} for {est_name}."
                    )

    def _get_step_name(
        self, name: Optional[str], step: Union[EstimatorLike, str]
    ) -> str:
        """Get the name of a step, with a count if it is repeated.

        Parameters
        ----------
        name : str, optional
            The name of the step, by default None.
        step : EstimatorLike or str
            The step to get the name for.

        Returns
        -------
        name : str
            The name of the step.

        """
        out = name
        if name is None:
            name = (
                step
                if isinstance(step, str)
                else step.__class__.__name__.lower()
            )
            count = np.array(
                [_step.name == name for _step in self._steps]
            ).sum()
            out = f"{name}_{count}" if count > 0 else name
        return out

    def _validate_step(
        self, step: Union[EstimatorLike, str], apply_to: ColumnTypesLike
    ) -> None:
        """Validate a step.

        Parameters
        ----------
        step : EstimatorLike or str
            The step to validate.
        apply_to : str
            The type of data the step is applied to.

        Raises
        ------
        ValueError
            If the step is not a valid step, if the transformer is added after
            adding a model, or if a transformer is added after a target
            transformer.

        """
        if self._is_transfromer_step(step):
            if self._added_model:
                raise_error("Cannot add a transformer after adding a model")
            if self._added_target_transformer and not self._is_model_step(
                step
            ):
                raise_error(
                    "Only a model can be added after a target transformer."
                )
            if apply_to == "target":
                self._added_target_transformer = True
        elif self._is_model_step(step):
            self._added_model = True
        else:
            raise_error(f"Cannot add a {step}. I don't know what it is.")

    def _check_X_types(
        self, X_types: Optional[Dict] = None  # noqa: N803
    ) -> Dict[str, List[str]]:
        """Check the X_types against the pipeline creator settings.

        Parameters
        ----------
        X_types : dict, optional
            The types of the columns in the data.

        Returns
        -------
        X_types : dict
            The types of the columns in the data after the check.

        Raises
        ------
        ValueError
            If there are extra types in the X_types that are not needed / used
            by the pipeline.

        Warns
        -----
        RuntimeWarning
            If there are extra types in the pipeline that are not specified in
            the X_types.

        """
        if X_types is None:
            X_types = {}
        # Get the set of all types in the X_types
        if X_types == {}:
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
        available_types = {*all_X_types, "*", ".*", "target", "continuous"}
        for needed_type in needed_types:
            if needed_type not in available_types:
                warn_with_log(
                    f"{needed_type} is not in the provided X_types={X_types}. "
                    "Make sure your pipeline has a transformer that creates "
                    "this type."
                )

        self.wrap = needed_types != {"continuous"}
        return X_types

    @staticmethod
    def _is_transfromer_step(step: Union[str, EstimatorLike]) -> bool:
        """Check if a step is a transformer."""
        if step in list_transformers():
            return True
        if hasattr(step, "fit") and hasattr(step, "transform"):
            return True
        return False

    @staticmethod
    def _is_model_step(step: Union[EstimatorLike, str]) -> bool:
        """Check if a step is a model."""
        if step in list_models():
            return True
        if hasattr(step, "fit") and hasattr(step, "predict"):
            return True
        return False

    @staticmethod
    def _wrap_step(
        name, step, column_types, row_select_col_type, row_select_vals
    ) -> JuColumnTransformer:
        """Wrap a step in a JuColumnTransformer.

        Parameters
        ----------
        name : str
            The name of the step.
        step : EstimatorLike
            The step to wrap.
        column_types : ColumnTypesLike
            The types of the columns the step is applied to.
        row_select_col_type : str or list of str or set of str or ColumnTypes
            The column types needed to select rows (default None).
        row_select_vals : str, int, bool or list of str, int, bool
            The value(s) which should be selected in the
            ``row_select_col_type`` to select the rows used for training
            (default None).

        """
        return JuColumnTransformer(
            name,
            step,
            column_types,
            row_select_col_type=row_select_col_type,
            row_select_vals=row_select_vals,
        )

    @staticmethod
    def _get_estimator_from(
        name: str, problem_type: str, **kwargs: Any
    ) -> EstimatorLike:
        """Get an estimator from a name.

        Parameters
        ----------
        name : str
            The name of the estimator.
        problem_type : str
            The problem type.
        **kwargs : dict
            The keyword arguments to pass to the estimator constructor.

        Returns
        -------
        estimator : EstimatorLike
            The estimator.

        Raises
        ------
        ValueError
            If the name is not a registered transformer or model.

        """
        if name in list_transformers():
            return get_transformer(name, **kwargs)
        if name in list_models():
            return get_model(name, problem_type, **kwargs)
        raise_error(
            f"{name} is neither a registered transformer"
            "nor a registered model."
        )


def _prepare_hyperparameter_tuning(
    params_to_tune: Union[Dict[str, Any], List[Dict[str, Any]]],
    search_params: Optional[Dict[str, Any]],
    pipeline: Pipeline,
):
    """Prepare hyperparameter tuning in the pipeline.

    Parameters
    ----------
    params_to_tune : dict
        A dictionary with the parameters to tune. The keys of the
        dictionary should be named 'STEP__PARAMETER', to be used as
        PARAMETER for STEP in the pipeline. Example:
        'svm__probability': True will set the parameter 'probability' of
        the 'svm' step. The value of the parameter must be a list of
        values to test.

    search_params : dict
        The parameters for the search. The following keys are accepted:

        * 'kind': The kind of search algorithm to use e.g.:
            'grid' or 'random'. All valid julearn searchers can be entered.
        * 'cv': If search is going to be used, the cross-validation
            splitting strategy to use. Defaults to same CV as for the model
            evaluation.
        * 'scoring': If search is going to be used, the scoring metric to
            evaluate the performance.

    pipeline : sklearn.pipeline.Pipeline
        The pipeline to apply/tune the hyperparameters

    Returns
    -------
    sklearn.pipeline.Pipeline
        The modified pipeline

    """
    logger.info("= Model Parameters =")

    search_params = prepare_search_params(search_params)
    if len(params_to_tune) > 0:
        search = search_params.pop("kind", "grid")
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
                warn_with_log(f"{search} is not a registered searcher. ")
                logger.info(
                    f"Tuning hyperparameters using not registered {search}"
                )

        if isinstance(params_to_tune, dict):
            logger.info("Hyperparameters:")
            for k, v in params_to_tune.items():
                logger.info(f"\t{k}: {v}")
        else:
            logger.info("Hyperparameters list:")
            for i_list, t_params in enumerate(params_to_tune):
                logger.info(f"\tSet {i_list}")
                for k, v in t_params.items():
                    logger.info(f"\t\t{k}: {v}")

        cv_inner = check_cv(cv_inner)  # type: ignore
        logger.info(f"Using inner CV scheme {cv_inner}")
        search_params["cv"] = cv_inner
        logger.info("Search Parameters:")
        for k, v in search_params.items():
            logger.info(f"\t{k}: {v}")

        # TODO: missing searcher typing
        pipeline = search(  # type: ignore
            pipeline, params_to_tune, **search_params
        )

    logger.info("====================")
    logger.info("")
    return pipeline
