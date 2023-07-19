"""TargetPipelineCreator class."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL

from typing import Optional, Union

import numpy as np

from ..transformers import get_transformer
from ..transformers.target import (
    get_target_transformer,
    list_target_transformers,
)
from ..utils.typing import EstimatorLike
from .target_pipeline import JuTargetPipeline


class TargetPipelineCreator:
    """TargetPipelineCreator class.

    Analogous to the PipelineCreator class, this class allows to create
    :class:`julearn.pipeline.target_pipeline.JuTargetPipeline` objects in an
    easy way.
    """

    def __init__(self) -> None:
        self._steps = []

    def add(
        self, step: str, name: Optional[str] = None, **params
    ) -> "TargetPipelineCreator":
        """Add a step to the pipeline.

        Parameters
        ----------
        step : str
            The step to add to the pipeline.
        name : str, optional
            The name of the step. If None, the name will be obtained from
            the step (default is None).
        **params
            Parameters for the step. This will mostly include
            hyperparameters or any other parameter for initialization.
        """
        # If the user did not give a name, we will create one.
        if name is None:
            name = (
                step
                if isinstance(step, str)
                else step.__class__.__name__.lower()
            )
        name = self._get_step_name(name, step)
        if step in list_target_transformers():
            estimator = get_target_transformer(step, **params)
        else:
            estimator = get_transformer(step, **params)
        self._steps.append((name, estimator))
        return self

    def to_pipeline(self) -> JuTargetPipeline:
        """Create a pipeline from the steps.

        Returns
        -------
        out : JuTargetPipeline
            The pipeline object.
        """
        return JuTargetPipeline(self._steps)

    def _get_step_name(
        self, name: Optional[str], step: Union[EstimatorLike, str]
    ) -> str:
        """Get the name of a step, with a count if it is repeated.

        Parameters
        ----------
        step : EstimatorLike or str
            The step to get the name for.

        Returns
        -------
        name : str
            The name of the step.
        """
        if name is None:
            name = (
                step
                if isinstance(step, str)
                else step.__class__.__name__.lower()
            )
        count = np.array([_step[0] == name for _step in self._steps]).sum()
        return f"{name}_{count}" if count > 0 else name

    def __str__(self) -> str:
        """Get a string representation of the TargetPipelineCreator."""
        out = "TargetPipelineCreator:\n"
        for i_step, step in enumerate(self._steps):
            out += f"  Step {i_step}: {step[0]}\n"
            out += f"    estimator:     {step[1]}\n"
        return out
