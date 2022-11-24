import warnings
from julearn.pipeline import PipelineCreator
from julearn.transformers import get_transformer
from julearn.estimators import get_model
from sklearn.compose import ColumnTransformer
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
    preprocess = [preprocess] if isinstance(preprocess, str) else preprocess
    for preprocess, step in zip(preprocess, pipeline.steps[1:-1]):
        name, transformer = step
        assert name.startswith(f"wrapped_{preprocess}")
        assert isinstance(transformer, ColumnTransformer)
        assert isinstance(
            transformer.transformers[0][1],
            get_transformer(preprocess).__class__)

    # check model step
    model_name, model = pipeline.steps[-1]
    # assert model_name == model
    assert isinstance(
        model,
        get_model(model_name,
                  problem_type=problem_type,
                  ).__class__
    )


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
            pipeline_creator.X_types_to_patterns(X_types)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            pipeline_creator.X_types_to_patterns(X_types)


@pytest.mark.parametrize(
    "apply_to,X_types,error",
    [
        (["continuous"], dict(cat="B"), True),
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
            pipeline_creator.X_types_to_patterns(X_types)
    else:
        pipeline_creator.X_types_to_patterns(X_types)
