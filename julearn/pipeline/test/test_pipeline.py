from julearn.pipeline import PipelineCreator
from julearn.transformers import get_transformer
from julearn.estimators import get_model
from sklearn.compose import ColumnTransformer
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
    "X,y,model,preprocess,problem_type",
    [pytest.lazy_fixture(
        ["X_multi_typed_iris", "y_typed_iris",
         "models_all_problem_types", "preprocessing", "all_problem_types"])],
)
def test_fit_and_transform_no_error(
        X, y, model, preprocess, problem_type
):

    pipeline = (PipelineCreator.from_list(preprocess, model_params={})
                .add(model,
                     problem_type=problem_type,)
                .to_pipeline(dict())
                )
    pipeline.fit(X, y)
    pipeline[:-1].transform(X)
