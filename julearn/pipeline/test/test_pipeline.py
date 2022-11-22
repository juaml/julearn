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
    pipeline = (PipelineCreator.from_list(preprocess, model_params={})
                .add(model,
                     problem_type=problem_type,)
                .to_pipeline(dict(), search_params={})
                )

    # check preprocessing steps
    # ignoring first step for types and last for model
    preprocess = [preprocess] if isinstance(preprocess, str) else preprocess
    for preprocess, step in zip(preprocess, pipeline.steps[1:-1]):
        name, transformer = step
        print(name, preprocess, type(preprocess))
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
    X_iris, y_iris, X_types_iris, model, preprocess,
    problem_type, get_default_params,
):

    preprocess = [preprocess] if isinstance(preprocess, str) else preprocess

    pipeline_creator = PipelineCreator()
    param_grid = {}

    used_types = (["continuous"]
                  if X_types_iris is None
                  else list(X_types_iris.keys()))
    wrap = False if used_types == ["continuous"] else True
    for step in preprocess:
        default_params = get_default_params(step)
        pipeline_creator = pipeline_creator.add(
            step, apply_to="continuous", **default_params
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
