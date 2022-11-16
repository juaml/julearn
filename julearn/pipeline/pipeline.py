from sklearn.pipeline import Pipeline
from .preprocessing import PreprocessCreator
from ..estimators import get_model
from ..prepare import prepare_model_params


def create_preprocessing(preprocess):
    # TODO validate Input
    return (
        (preprocess
         if isinstance(preprocess, PreprocessCreator)
         else PreprocessCreator.from_list(preprocess)
         )
    )


def create_mode_step(model, problem_type=None):

    if isinstance(model, str):
        model_name = model
        model = get_model(model, problem_type=problem_type)
    elif hasattr(model, "fit") and hasattr(model, "predict"):
        model_name = model.__name__
    else:
        raise ValueError("TODO ADD Error")  # TODO ADD error
    return (model_name, model)


def create_pipeline(preprocessor, model_step):
    steps = preprocessor.steps
    steps.append(model_step)
    return Pipeline(steps).set_output(transform="pandas")


def create_cv_pipeline(
    model,
    problem_type=None,
    preprocess=None,
    model_params=None,
):

    # TODO: validate input
    preprocessor = (None if preprocess is None
                    else create_preprocessing(preprocess)
                    )

    model_step = create_mode_step(model, problem_type)

    pipeline = create_pipeline(
        preprocessor=preprocessor,
        model_step=model_step
    )

    # hyperparameter grid
    param_grid = {} if model_params is None else model_params
    param_grid = {**param_grid, **preprocessor.param_grid}

    # deal with cross validation (cv)
    return prepare_model_params(param_grid, pipeline)
