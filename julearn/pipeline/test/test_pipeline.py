import warnings
from julearn.pipeline import PipelineCreator
from julearn.pipeline.pipeline import JuColumnTransformer
from julearn.transformers import get_transformer
from julearn.models import get_model
from sklearn.model_selection import GridSearchCV
import pytest


@pytest.mark.parametrize(
    "model,preprocess,problem_type",
    [pytest.lazy_fixture(
        ["models_all_problem_types", "preprocessing", "all_problem_types"])],
)
def test_construction_working(model, preprocess, problem_type
                              ):
    pipeline_creator = PipelineCreator()
    preprocess = preprocess if isinstance(preprocess, list) else [preprocess]
    for step in preprocess:
        pipeline_creator.add(step, apply_to="categorical")
    pipeline = (pipeline_creator
                .add(model, problem_type=problem_type,)
                .to_pipeline(dict(categorical=["A"]), search_params={})
                )

    # check preprocessing steps
    # ignoring first step for types and last for model
    for _preprocess, step in zip(preprocess, pipeline.steps[1:-1]):
        name, transformer = step
        assert name.startswith(f"wrapped_{_preprocess}")
        assert isinstance(transformer, JuColumnTransformer)
        assert isinstance(
            transformer.transformer,
            get_transformer(_preprocess).__class__)

    # check model step
    model_name, model = pipeline.steps[-1]
    assert isinstance(
        model.model,
        get_model(model_name.replace("wrapped_", ""),
                  problem_type=problem_type,
                  ).__class__
    )
    assert len(preprocess) + 2 == len(pipeline.steps)


@pytest.mark.parametrize(
    "model,preprocess,problem_type",
    [pytest.lazy_fixture(
        ["models_all_problem_types", "preprocessing", "all_problem_types"]
    )],
)
def test_fit_and_transform_no_error(
        X_iris, y_iris, model, preprocess, problem_type
):

    pipeline = (PipelineCreator.from_list(preprocess, model_params={})
                .add(model,
                     problem_type=problem_type,)
                .to_pipeline(dict())
                )
    pipeline.fit(X_iris, y_iris)
    pipeline[:-1].transform(X_iris)


@pytest.mark.parametrize(
    "model,preprocess,problem_type",
    [pytest.lazy_fixture(
        ["models_all_problem_types",
         "preprocessing",
         "all_problem_types"
         ]),

     ],
)
def test_hyperparameter_tuning(
    X_types_iris, model, preprocess,
    problem_type, get_default_params,
):

    preprocess = [preprocess] if isinstance(preprocess, str) else preprocess

    pipeline_creator = PipelineCreator()
    param_grid = {}

    used_types = (["continuous"]
                  if X_types_iris in [None, dict()]
                  else list(X_types_iris.keys()))
    wrap = False if used_types == ["continuous"] else True
    for step in preprocess:
        default_params = get_default_params(step)
        pipeline_creator = pipeline_creator.add(
            step, apply_to=used_types, **default_params
        )
        name = f"wrapped_{step}__{step}" if wrap else step
        params = {f"{name}__{param}": val
                  for param, val in default_params.items()
                  }
        param_grid.update(params)

    model_params = get_default_params(model)
    pipeline_creator = pipeline_creator.add(
        model, problem_type=problem_type,
        **model_params
    )
    model = f"wrapped_{model}__{model}" if wrap else model
    param_grid.update({f"{model}__{param}": val
                       for param, val in model_params.items()})
    pipeline = pipeline_creator.to_pipeline(X_types=X_types_iris)

    assert isinstance(pipeline, GridSearchCV)
    assert pipeline.param_grid == param_grid


@pytest.mark.parametrize(
    "apply_to,X_types,warns",
    [
        (["continuous"], dict(continuous="A", cat="B"), True),
        (["continuous", "cat"], dict(continuous="A", cat="B"), False),
        ("*", dict(continuous="A", cat="B"), False),
        ([".*"], dict(continuous="A", cat="B"), False),
    ]
)
def test_X_types_to_patter_warnings(apply_to, X_types, warns):
    pipeline_creator = (
        PipelineCreator()
        .add("zscore", apply_to=apply_to)
    )
    if warns:
        with pytest.warns(
            match=".* is provided but never used by a transformer. "
        ):
            pipeline_creator.check_X_types(X_types)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            pipeline_creator.check_X_types(X_types)


@pytest.mark.parametrize(
    "apply_to,X_types,error",
    [
        (["duck"], dict(), True),
        (["duck"], dict(duck="B"), False),
        (["continuous"], dict(), False),
        (["continuous", "cat"], dict(continuous="A", cat="B"), False),
        ("*", dict(continuous="A", cat="B"), False),
        ([".*"], dict(continuous="A", cat="B"), False),
    ]
)
def test_X_types_to_patter_errors(apply_to, X_types, error):
    pipeline_creator = (
        PipelineCreator()
        .add("zscore", apply_to=apply_to)
    )
    if error:
        with pytest.raises(ValueError,
                           match=".* is not in the provided X_types="):
            pipeline_creator.check_X_types(X_types)
    else:
        pipeline_creator.check_X_types(X_types)


def test_model_uses_param():

    pipeline_creator = (
        PipelineCreator()
        .add("rf", apply_to="chicken")
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pipeline_creator.check_X_types(dict(chicken="teriyaki"))

    with pytest.warns(
        match=".* is provided but never used by a transformer. "
    ):
        pipeline_creator.check_X_types(
            dict(chicken="teriyaki", duck="B"))
    with pytest.raises(ValueError,
                       match=".* is not in the provided X_types="):
        pipeline_creator.check_X_types(dict(duck="teriyaki"))


def test_added_model_target_transform():
    pipeline_creator = PipelineCreator().add("zscore", apply_to="continuous")
    assert pipeline_creator._added_target_transformer is False
    pipeline_creator.add("zscore", apply_to="target")
    assert pipeline_creator._added_target_transformer
    assert pipeline_creator._added_model is False
    pipeline_creator.add("rf")
    assert pipeline_creator._added_model


def test_stacking(X_iris, y_iris):
    # Define our feature types
    X_types = {
        "sepal": ["sepal_length", "sepal_width"],
        "petal": ["petal_length", "petal_width"],
    }
    # Create the pipeline for the sepal features
    model_sepal = PipelineCreator()
    model_sepal.add("filter_columns", apply_to="*", keep="sepal")
    model_sepal.add("zscore", apply_to="*")
    model_sepal.add("svm", apply_to="*")

    # Create the pipeline for the petal features
    model_petal = PipelineCreator()
    model_petal.add("filter_columns", apply_to="*", keep="petal")
    model_petal.add("zscore", apply_to="*")
    model_petal.add("rf", apply_to="*")

    # Create the stacking model
    model = PipelineCreator()
    model.add(
        "stacking",
        estimators=[[("sepal", model_sepal), ("petal", model_petal)]],
        apply_to="*"
    )
    model = model.to_pipeline(X_types)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model.fit(X_iris, y_iris)
