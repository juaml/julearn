from julearn.pipeline import create_cv_pipeline
from julearn.transformers import get_transformer
from julearn.estimators import get_model
from sklearn.compose import ColumnTransformer


def test_construction_working(
        models_all_problem_types, preprocessing,
        all_problem_types
):
    pipeline = create_cv_pipeline(
        model=models_all_problem_types,
        problem_type=all_problem_types,
        preprocess=preprocessing
    )

    # check preprocessing steps
    for preprocess, step in zip(preprocessing, pipeline.steps[:-1]):
        name, transformer = step
        assert name.startswith(f"wrapped_{preprocess}")
        assert isinstance(transformer, ColumnTransformer)
        assert isinstance(
            transformer.transformers[0][1],
            get_transformer(preprocess).__class__)

    # check model step
    model_name, model = pipeline.steps[-1]
    assert model_name == models_all_problem_types
    assert isinstance(
        model,
        get_model(models_all_problem_types,
                  problem_type=all_problem_types,
                  ).__class__
    )


def test_fit_and_transform_no_error(
        X_typed_iris, y_typed_iris,
        models_all_problem_types, preprocessing,
        all_problem_types
):
    pipeline = create_cv_pipeline(
        model=models_all_problem_types,
        problem_type=all_problem_types,
        preprocess=preprocessing
    )

    pipeline.fit(X_typed_iris, y_typed_iris)
    pipeline[:-1].transform(X_typed_iris)


def test_construction_hyperparams():
    ...
