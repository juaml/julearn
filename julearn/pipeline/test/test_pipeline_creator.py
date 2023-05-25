"""Provides tests for the pipeline creator module."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL

import warnings
from typing import Callable, Dict, List

import pandas as pd
import pytest
from pytest_lazyfixture import lazy_fixture
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from julearn.base import ColumnTypesLike, WrapModel
from julearn.models import get_model
from julearn.pipeline import PipelineCreator, TargetPipelineCreator
from julearn.pipeline.pipeline_creator import JuColumnTransformer
from julearn.transformers import get_transformer


@pytest.mark.parametrize(
    "model,preprocess,problem_type",
    [
        lazy_fixture(
            ["models_all_problem_types", "preprocessing", "all_problem_types"]
        )
    ],
)
def test_construction_working(
    model: str, preprocess: List[str], problem_type: str
) -> None:
    """Test that the pipeline constructions works as expected.

    Parameters
    ----------
    model : str
        The model to test.
    preprocess : List[str]
        The preprocessing steps to test.
    problem_type : str
        The problem type to test.
    """
    creator = PipelineCreator(problem_type=problem_type)
    preprocess = preprocess if isinstance(preprocess, list) else [preprocess]
    for step in preprocess:
        creator.add(step, apply_to="categorical")
    creator.add(model)
    X_types = {"categorical": ["A"]}
    pipeline = creator.to_pipeline(X_types=X_types)

    # check preprocessing steps
    # ignoring first step for types and last for model
    for element in zip(preprocess, pipeline.steps[1:-1]):
        _preprocess, (name, transformer) = element
        assert name.startswith(f"{_preprocess}")
        assert isinstance(transformer, JuColumnTransformer)
        assert isinstance(
            transformer.transformer, get_transformer(_preprocess).__class__
        )

    # check model step
    model_name, model = pipeline.steps[-1]
    assert isinstance(model, WrapModel)
    assert isinstance(
        model.model,
        get_model(
            model_name,
            problem_type=problem_type,
        ).__class__,
    )
    assert len(preprocess) + 2 == len(pipeline.steps)


@pytest.mark.parametrize(
    "model,preprocess,problem_type",
    [
        lazy_fixture(
            ["models_all_problem_types", "preprocessing", "all_problem_types"]
        )
    ],
)
def test_fit_and_transform_no_error(
    X_iris: pd.DataFrame,
    y_iris: pd.Series,
    model: str,
    preprocess: List[str],
    problem_type: str,
) -> None:
    """Test that the pipeline fit and transform does not give an error.

    Parameters
    ----------
    X_iris : pd.DataFrame
        The iris dataset features
    y_iris : pd.Series
        The iris dataset target variable.
    model : str
        The model to test.
    preprocess : List[str]
        The preprocessing steps to test.
    problem_type : str
        The problem type to test.
    """
    creator = PipelineCreator.from_list(
        preprocess, model_params={}, problem_type=problem_type
    )
    creator.add(model)
    pipeline = creator.to_pipeline({})
    pipeline.fit(X_iris, y_iris)
    pipeline[:-1].transform(X_iris)


@pytest.mark.parametrize(
    "model,preprocess,problem_type",
    [
        lazy_fixture(
            ["models_all_problem_types", "preprocessing", "all_problem_types"]
        ),
    ],
)
def test_hyperparameter_tuning(
    X_types_iris: Dict[str, List[str]],
    model: str,
    preprocess: List[str],
    problem_type: str,
    get_tuning_params: Callable,
    search_params: Dict[str, List],
) -> None:
    """Test that the pipeline hyperparameter tuning works as expected.

    Parameters
    ----------
    X_types_iris : Dict[str, List[str]]
        The iris dataset features types.
    model : str
        The model to test.
    preprocess : List[str]
        The preprocessing steps to test.
    problem_type : str
        The problem type to test.
    get_tuning_params : Callable
        A function that returns the tuning hyperparameters for a given step.
    """
    if isinstance(preprocess, str):
        preprocess = [preprocess]

    creator = PipelineCreator(problem_type=problem_type)
    param_grid = {}

    used_types = (
        ["continuous"]
        if X_types_iris in [None, dict()]
        else list(X_types_iris.keys())
    )
    for step in preprocess:
        default_params = get_tuning_params(step)
        creator.add(step, apply_to=used_types, **default_params)
        params = {
            f"{step}__{param}": val for param, val in default_params.items()
        }
        param_grid.update(params)

    model_params = get_tuning_params(model)
    creator.add(model, **model_params)

    param_grid.update(
        {f"{model}__{param}": val for param, val in model_params.items()}
    )
    pipeline = creator.to_pipeline(
        X_types=X_types_iris, search_params=search_params
    )

    kind = "grid"
    if search_params is not None:
        kind = search_params.get("kind", "grid")
    if kind == "grid":
        assert isinstance(pipeline, GridSearchCV)
        assert pipeline.param_grid == param_grid
    else:
        assert isinstance(pipeline, RandomizedSearchCV)
        assert pipeline.param_distributions == param_grid


@pytest.mark.parametrize(
    "X_types,apply_to,warns",
    [
        ({"duck": "B"}, ["duck", "chicken"], True),
        ({"duck": "B"}, ["duck"], False),
        ({}, ["continuous"], False),
        (None, ["continuous"], False),
        ({"continuous": "A", "cat": "B"}, ["continuous", "cat"], False),
        ({"continuous": "A"}, ["continuous", "target"], False),
        ({"continuous": "A", "cat": "B"}, ["*"], False),
        ({"continuous": "A", "cat": "B"}, [".*"], False),
    ],
)
def test_X_types_to_pattern_warnings(
    X_types: Dict[str, List[str]], apply_to: ColumnTypesLike, warns: bool
) -> None:
    """Test that the X_types raises the expected warnings.

    Parameters
    ----------
    X_types : Dict[str, List[str]]
        The X_types to test.
    apply_to : ColumnTypesLike
        The apply_to to test.
    warns : bool
        Whether the test should raise a warning.
    """
    pipeline_creator = PipelineCreator(problem_type="classification").add(
        "zscore", apply_to=apply_to
    )
    if warns:
        with pytest.warns(match="is not in the provided X_types"):
            pipeline_creator._check_X_types(X_types)
    else:
        pipeline_creator._check_X_types(X_types)


@pytest.mark.parametrize(
    "X_types,apply_to,error",
    [
        ({}, ["duck"], True),
        ({"duck": "B"}, ["duck"], False),
        ({}, ["continuous"], False),
        (None, ["continuous"], False),
        ({"continuous": "A", "cat": "B"}, ["continuous", "cat"], False),
        ({"continuous": "A", "cat": "B"}, ["continuous"], True),
        ({"continuous": "A", "cat": "B"}, ["*"], False),
        ({"continuous": "A", "cat": "B"}, [".*"], False),
    ],
)
def test_X_types_to_pattern_errors(
    X_types: Dict[str, List[str]], apply_to: ColumnTypesLike, error: bool
) -> None:
    """Test that the X_types raises the expected errors.

    Parameters
    ----------
    X_types : Dict[str, List[str]]
        The X_types to test.
    apply_to : ColumnTypesLike
        The apply_to to test.
    error : bool
        Whether the test should raise a warning.
    """
    pipeline_creator = PipelineCreator(problem_type="classification").add(
        "zscore", apply_to=apply_to
    )
    if error:
        with pytest.raises(ValueError, match="Extra X_types were provided"):
            pipeline_creator._check_X_types(X_types)
    else:
        pipeline_creator._check_X_types(X_types)


def test_pipelinecreator_default_apply_to() -> None:
    """Test pipeline creator using the default apply_to."""

    pipeline_creator = PipelineCreator(problem_type="classification").add(
        "rf", apply_to="chicken"
    )

    with pytest.raises(ValueError, match="Extra X_types were provided"):
        pipeline_creator._check_X_types({"duck": "B"})

    pipeline_creator = PipelineCreator(problem_type="classification").add(
        "rf", apply_to=["chicken", "duck"]
    )
    with pytest.warns(match="is not in the provided X_types"):
        pipeline_creator._check_X_types({"chicken": "teriyaki"})

    pipeline_creator = PipelineCreator(problem_type="classification").add(
        "rf", apply_to="*"
    )
    pipeline_creator._check_X_types({"duck": "teriyaki"})


def test_pipelinecreator_default_constructor_apply_to() -> None:
    """Test pipeline creator using a default apply_to in the constructor."""
    pipeline_creator = PipelineCreator(
        problem_type="classification", apply_to="duck"
    ).add("rf")
    pipeline_creator._check_X_types({"duck": "teriyaki"})

    pipeline_creator = PipelineCreator(
        problem_type="classification", apply_to="duck"
    )
    pipeline_creator.add("zscore", apply_to="chicken")
    pipeline_creator.add("rf")
    pipeline_creator._check_X_types({"duck": "teriyaki", "chicken": "1"})


def test_added_model_target_transform() -> None:
    """Test that the added model and target transformer are set correctly."""
    pipeline_creator = PipelineCreator(problem_type="classification").add(
        "zscore", apply_to="continuous"
    )
    assert pipeline_creator._added_target_transformer is False
    pipeline_creator.add("zscore", apply_to="target")
    assert pipeline_creator._added_target_transformer
    assert pipeline_creator._added_model is False
    pipeline_creator.add("rf")
    assert pipeline_creator._added_model


def test_stacking(X_iris: pd.DataFrame, y_iris: pd.Series) -> None:
    """Test that the stacking model works correctly."""
    # Define our feature types
    X_types = {
        "sepal": ["sepal_length", "sepal_width"],
        "petal": ["petal_length", "petal_width"],
    }
    # Create the pipeline for the sepal features
    model_sepal = PipelineCreator(problem_type="classification", apply_to="*")
    model_sepal.add("filter_columns", keep="sepal")
    model_sepal.add("zscore")
    model_sepal.add("svm")

    # Create the pipeline for the petal features
    model_petal = PipelineCreator(problem_type="classification", apply_to="*")
    model_petal.add("filter_columns", keep="petal")
    model_petal.add("zscore")
    model_petal.add("rf")

    # Create the stacking model
    model = PipelineCreator(problem_type="classification")
    model.add(
        "stacking",
        estimators=[[("sepal", model_sepal), ("petal", model_petal)]],
        apply_to="*",
    )
    model = model.to_pipeline(X_types)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model.fit(X_iris, y_iris)


def test_added_repeated_transformers() -> None:
    """Test that the repeated transformers names are set correctly."""
    pipeline_creator = PipelineCreator(problem_type="classification")
    pipeline_creator.add("zscore", apply_to="continuous")
    pipeline_creator.add("zscore", apply_to="duck")
    pipeline_creator.add("rf")
    assert len(pipeline_creator._steps) == 3
    assert pipeline_creator._steps[0].name == "zscore"
    assert pipeline_creator._steps[1].name == "zscore_1"


# TODO: Identify what are we testing here
# def test_hyperparameter_ter() -> None:
#     PipelineCreator(problem_type="classification").add(
#         "confound_removal", model_confound=RandomForestRegressor()
#     )


def test_target_pipe(X_iris, y_iris) -> None:
    """Test that the target pipeline works correctly."""
    X_types = {
        "continuous": ["sepal_length", "sepal_width", "petal_length"],
        "confounds": ["petal_width"],
    }
    target_pipeline = TargetPipelineCreator().add(
        "confound_removal", confounds=["confounds", "continuous"]
    )
    pipeline_creator = (
        PipelineCreator(problem_type="regression")
        .add(target_pipeline, apply_to="target")
        .add("svm", C=[1, 2])
    )
    pipe = pipeline_creator.to_pipeline(
        X_types, search_params=dict(kind="random")
    )
    pipe.fit(X_iris, y_iris)


def test_raise_wrong_problem_type() -> None:
    """Test that the correct error is raised when the problem type is wrong."""
    with pytest.raises(ValueError, match="`problem_type` should"):
        PipelineCreator(problem_type="binary")


def test_raise_wrong_problem_type_added_to_step() -> None:
    """Test error when problem type is passed to a step."""
    with pytest.raises(ValueError, match="Please provide the problem_type"):
        PipelineCreator(problem_type="classification").add(
            "svm", problem_type="classification"
        )


def test_raise_error_not_target_pipe() -> None:
    """Test error when target pipeline is not applied to target."""
    with pytest.raises(ValueError, match="TargetPipelineCreator can"):
        target_pipeline = TargetPipelineCreator().add(
            "confound_removal", confounds=["confounds", "continuous"]
        )
        PipelineCreator(problem_type="regression").add(
            target_pipeline, apply_to="confounds"
        )


def test_raise_pipe_no_model() -> None:
    """Test error when no model is added to the pipeline."""
    X_types = {
        "continuous": [
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width",
        ],
    }
    pipeline_creator = PipelineCreator(problem_type="regression").add("zscore")
    with pytest.raises(ValueError, match="Cannot create a pipe"):
        pipeline_creator.to_pipeline(X_types)


def test_raise_pipe_wrong_searcher() -> None:
    """Test error when the searcher is not a valid julearn searcher."""
    X_types = {
        "continuous": [
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width",
        ],
    }
    pipeline_creator = PipelineCreator(problem_type="regression").add(
        "svm", C=[1, 2]
    )
    with pytest.raises(
        ValueError,
        match="The searcher no_search is not a valid julearn searcher",
    ):
        pipeline_creator.to_pipeline(
            X_types, search_params=dict(kind="no_search")
        )


def test_PipelineCreator_repeated_steps() -> None:
    """Test the pipeline creator with repeated steps."""

    # Without explicit naming, it should not be considered repeated
    creator = PipelineCreator(problem_type="classification")
    creator.add("zscore", apply_to="continuous")
    creator.add("zscore", apply_to="continuous")
    creator.add("rf")
    assert len(creator._steps) == 3
    assert creator._steps[0].name == "zscore"
    assert creator._steps[1].name == "zscore_1"

    # With explicit naming, it should be considered repeated
    creator2 = PipelineCreator(problem_type="classification")
    creator2.add("zscore", name="scale", apply_to="continuous")
    creator2.add("zscore", name="scale", apply_to="continuous")
    creator2.add("rf")
    assert len(creator2._steps) == 3
    assert creator2._steps[0].name == "scale"
    assert creator2._steps[1].name == "scale"


def test_PipelineCreator_repeated_steps_error() -> None:
    """Test error with repeated steps."""

    # With explicit naming, it should be considered repeated
    creator = PipelineCreator(problem_type="classification")
    creator.add("zscore", name="scale", apply_to="continuous")
    creator.add("pca", name="pca", apply_to="continuous")
    with pytest.raises(ValueError, match="Repeated step names are only"):
        creator.add("scaler_robust", name="scale", apply_to="continuous")


def test_PipelineCreator_split() -> None:
    """Test the pipeline creator split."""
    # No repetition, split should create one pipeline
    creator1 = PipelineCreator(problem_type="classification")
    creator1.add("zscore", apply_to="continuous")
    creator1.add("zscore", apply_to="continuous")
    creator1.add("rf")
    assert len(creator1._steps) == 3
    assert creator1._steps[0].name == "zscore"
    assert creator1._steps[1].name == "zscore_1"

    out1 = creator1.split()
    assert len(out1) == 1
    assert len(out1[0]._steps) == 3
    assert out1[0]._steps[0].name == "zscore"
    assert out1[0]._steps[1].name == "zscore_1"
    assert out1[0]._steps[2].name == "rf"

    # Repeated a step twice, split should create two pipelines
    creator2 = PipelineCreator(problem_type="classification")
    creator2.add("zscore", name="scale", apply_to="continuous")
    creator2.add("zscore", name="scale", apply_to="continuous")
    creator2.add("rf")
    assert len(creator2._steps) == 3
    assert creator2._steps[0].name == "scale"
    assert creator2._steps[1].name == "scale"

    out2 = creator2.split()
    assert len(out2) == 2
    assert len(out2[0]._steps) == 2
    assert out2[0]._steps[0].name == "scale"
    assert out2[0]._steps[1].name == "rf"
    assert len(out2[1]._steps) == 2
    assert out2[1]._steps[0].name == "scale"
    assert out2[1]._steps[1].name == "rf"

    # Repeated a step three times, split should create three pipelines
    creator3 = PipelineCreator(problem_type="classification")
    creator3.add("zscore", name="scale", apply_to="continuous")
    creator3.add("zscore", name="scale", apply_to="continuous")
    creator3.add("scaler_robust", name="scale", apply_to="continuous")
    creator3.add("rf")
    assert len(creator3._steps) == 4
    assert creator3._steps[0].name == "scale"
    assert creator3._steps[1].name == "scale"
    assert creator3._steps[2].name == "scale"

    out3 = creator3.split()
    assert len(out3) == 3
    assert len(out3[0]._steps) == 2
    assert out3[0]._steps[0].name == "scale"
    assert out3[0]._steps[1].name == "rf"

    assert len(out3[1]._steps) == 2
    assert out3[1]._steps[0].name == "scale"
    assert out3[1]._steps[1].name == "rf"

    assert len(out3[2]._steps) == 2
    assert out3[2]._steps[0].name == "scale"
    assert out3[2]._steps[1].name == "rf"

    # Repeated two step twice, split should create 4 pipelines
    creator4 = PipelineCreator(problem_type="classification")
    creator4.add("zscore", name="scale", apply_to="continuous")
    creator4.add("scaler_robust", name="scale", apply_to="continuous")
    creator4.add("pca", apply_to="continuous")
    creator4.add("rf", name="model")
    creator4.add("svm", name="model")
    assert len(creator4._steps) == 5
    assert creator4._steps[0].name == "scale"
    assert creator4._steps[1].name == "scale"
    assert creator4._steps[2].name == "pca"
    assert creator4._steps[3].name == "model"
    assert creator4._steps[4].name == "model"

    out4 = creator4.split()
    assert len(out4) == 4
    for i in range(4):
        assert len(out4[i]._steps) == 3
        assert out4[i]._steps[0].name == "scale"
        assert out4[i]._steps[1].name == "pca"
        assert out4[i]._steps[2].name == "model"

    assert isinstance(out4[0]._steps[0].estimator, StandardScaler)
    assert isinstance(out4[0]._steps[2].estimator, RandomForestClassifier)

    assert isinstance(out4[1]._steps[0].estimator, StandardScaler)
    assert isinstance(out4[1]._steps[2].estimator, SVC)

    assert isinstance(out4[2]._steps[0].estimator, RobustScaler)
    assert isinstance(out4[2]._steps[2].estimator, RandomForestClassifier)

    assert isinstance(out4[3]._steps[0].estimator, RobustScaler)
    assert isinstance(out4[3]._steps[2].estimator, SVC)
