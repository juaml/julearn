from typing import List, Tuple, Union

from ..transformers.target import JuTargetTransformer
from ..utils.typing import TransformerLike


class TargetPipeline:
    def __init__(
        self,
        steps: List[Tuple[str, Union[JuTargetTransformer, TransformerLike]]],
    ):
        self.steps = steps

    def fit(self, X, y):
        for _, t_step in self.steps:
            if isinstance(t_step, JuTargetTransformer):
                y = t_step.fit_transform(X, y)
            else:
                y = t_step.fit_transform(y[:, None])[:, 0]
        return self

    def transform(self, X, y):
        for _, t_step in self.steps:
            if isinstance(t_step, JuTargetTransformer):
                y = t_step.transform(X, y)
            else:
                y = t_step.transform(y[:, None])[:, 0]
        return y
