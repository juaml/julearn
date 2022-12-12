from typing import Optional
import numpy as np

from ..transformers.target import (
    get_target_transformer,
    list_target_transformers,
)
from ..transformers import get_transformer

from .target_pipeline import JuTargetPipeline


class TargetPipelineCreator:  # Pipeline creator
    """TargetPipelineCreator class."""

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
        """
        # If the user did not give a name, we will create one.
        if name is None:
            name = (
                step
                if isinstance(step, str)
                else step.__class__.__name__.lower()
            )
            name = self._ensure_name(name)
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

    def _ensure_name(self, name: str) -> str:
        """Ensure that the name is unique.

        Parameters
        ----------
        name : str
            The name to check.

        Returns
        -------
        str
            The name with a number appended if it is not unique.
        """
        count = np.array([_step[0] == name for _step in self._steps]).sum()
        return f"{name}_{count}" if count > 0 else name
