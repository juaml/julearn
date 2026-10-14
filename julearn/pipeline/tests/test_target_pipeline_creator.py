"""Provides tests for the target creator module."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL

import pytest
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from julearn.pipeline import JuTargetPipeline, TargetPipelineCreator
from julearn.transformers.target import TargetConfoundRemover


def test_TargetPipelineCreator() -> None:
    """Test the target pipeline creator."""

    creator = TargetPipelineCreator()
    creator.add("zscore")
    creator.add("scaler_minmax")
    creator.add("confound_removal", confounds="confounds")
    pipeline = creator.to_pipeline()

    assert isinstance(pipeline, JuTargetPipeline)
    assert len(pipeline.steps) == 3
    assert isinstance(pipeline.steps[0][1], StandardScaler)
    assert isinstance(pipeline.steps[1][1], MinMaxScaler)
    assert isinstance(pipeline.steps[2][1], TargetConfoundRemover)


def test_TargetPipelineCreator_repeated_names() -> None:
    """Test the target pipeline creator."""

    creator = TargetPipelineCreator()
    creator.add("zscore")
    creator.add("zscore")
    pipeline = creator.to_pipeline()

    assert isinstance(pipeline, JuTargetPipeline)
    assert len(pipeline.steps) == 2
    assert pipeline.steps[0][0] == "zscore"
    assert pipeline.steps[1][0] == "zscore_1"


@pytest.mark.parametrize(
    "name,expected",
    [
        (None, "zscore"),
        ("scale", "scale"),
    ],
)
def test_TargetPipelineCreator_many_repeated_names(
    name: str | None, expected: str
) -> None:
    """Test that a step repeated more than twice gets unique names.

    Parameters
    ----------
    name : str or None
        The name to give to the step.
    expected : str
        The expected name of the first step.

    """
    creator = TargetPipelineCreator()
    for _ in range(4):
        creator.add("zscore", name=name)
    pipeline = creator.to_pipeline()

    names = [x[0] for x in pipeline.steps]
    assert names == [
        expected,
        f"{expected}_1",
        f"{expected}_2",
        f"{expected}_3",
    ]
