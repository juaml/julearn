"""Test the pipeline merger module."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>prepa
# License: AGPL

import pytest
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from julearn.pipeline import PipelineCreator
from julearn.pipeline.merger import merge_pipelines


def test_merger_pipelines() -> None:
    """Test the pipeline merger."""

    creator1 = PipelineCreator(problem_type="classification")
    creator1.add("zscore", name="scaler", apply_to="continuous")
    creator1.add("rf")

    creator2 = PipelineCreator(problem_type="classification")
    creator2.add("scaler_robust", name="scaler", apply_to="continuous")
    creator2.add("rf")

    pipe1 = creator1.to_pipeline()
    pipe2 = creator2.to_pipeline()

    merged = merge_pipelines(pipe1, pipe2, search_params=None)

    assert isinstance(merged, GridSearchCV)
    assert isinstance(merged.estimator, Pipeline)
    assert len(merged.estimator.named_steps) == 3
    named_steps = list(merged.estimator.named_steps.keys())
    assert "scaler" == named_steps[1]
    assert "rf" == named_steps[2]
    assert len(merged.param_grid) == 2

    search_params = {"kind": "random"}
    creator3 = PipelineCreator(problem_type="classification")
    creator3.add("zscore", name="scaler", apply_to="continuous")
    creator3.add("rf", max_features=[2, 3, 7, 42])
    pipe3 = creator3.to_pipeline(search_params=search_params)

    merged = merge_pipelines(pipe1, pipe2, pipe3, search_params=search_params)

    assert isinstance(merged, RandomizedSearchCV)
    assert isinstance(merged.estimator, Pipeline)
    assert len(merged.estimator.named_steps) == 3
    named_steps = list(merged.estimator.named_steps.keys())
    assert "scaler" == named_steps[1]
    assert "rf" == named_steps[2]
    assert len(merged.param_distributions) == 3
    assert merged.param_distributions[-1]["rf__max_features"] == [2, 3, 7, 42]


def test_merger_errors() -> None:
    """Test that the merger raises errors when it should."""
    creator1 = PipelineCreator(problem_type="classification")
    creator1.add("zscore", name="scaler", apply_to="continuous")
    creator1.add("rf")

    creator2 = PipelineCreator(problem_type="classification")
    creator2.add("scaler_robust", name="scaler", apply_to="continuous")
    creator2.add("rf", n_estimators=[10, 100])

    pipe1 = creator1.to_pipeline()
    pipe2 = creator2.to_pipeline(search_params={"kind": "grid"})

    with pytest.raises(ValueError, match="Only pipelines and searchers"):
        merge_pipelines(pipe1, SVC(), search_params=None)

    search_params = {"kind": "random"}

    with pytest.raises(
        ValueError,
        match="At least one of the pipelines to merge is a GridSearchCV",
    ):
        merge_pipelines(pipe1, pipe2, search_params=search_params)

    search_params = {"kind": "grid"}
    pipe2 = creator2.to_pipeline(search_params={"kind": "random"})

    with pytest.raises(
        ValueError,
        match="one of the pipelines to merge is a RandomizedSearchCV",
    ):
        merge_pipelines(pipe1, pipe2, search_params=search_params)

    pipe3 = GridSearchCV(SVC(), param_grid={"C": [1, 10]})
    with pytest.raises(
        ValueError,
        match="All searchers must use a pipeline.",
    ):
        merge_pipelines(pipe1, pipe3, search_params=None)

    creator4 = PipelineCreator(problem_type="classification")
    creator4.add("scaler_robust", name="scaler", apply_to="continuous")
    creator4.add("pca")
    creator4.add("rf", n_estimators=[10, 100])
    pipe4 = creator4.to_pipeline(search_params={"kind": "grid"})
    with pytest.raises(
        ValueError,
        match="must have the same named steps.",
    ):
        merge_pipelines(pipe1, pipe4, search_params=None)
