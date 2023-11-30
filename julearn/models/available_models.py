"""Provide registry of models."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL

from copy import deepcopy
from typing import Any, Dict, List, Optional, Type

from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    BaggingClassifier,
    BaggingRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    StackingClassifier,
    StackingRegressor,
)
from sklearn.gaussian_process import (
    GaussianProcessClassifier,
    GaussianProcessRegressor,
)
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
    LogisticRegressionCV,
    Ridge,
    RidgeClassifier,
    RidgeClassifierCV,
    RidgeCV,
    SGDClassifier,
    SGDRegressor,
)
from sklearn.naive_bayes import (
    BernoulliNB,
    CategoricalNB,
    ComplementNB,
    GaussianNB,
    MultinomialNB,
)
from sklearn.svm import SVC, SVR

from ..utils import logger, raise_error, warn_with_log
from ..utils.typing import ModelLike
from .dynamic import DynamicSelection


_available_models: Dict[str, Dict[str, Any]] = {
    # SVM
    "svm": {
        "regression": SVR,
        "classification": SVC,
    },
    # Ensemble
    "rf": {
        "regression": RandomForestRegressor,
        "classification": RandomForestClassifier,
    },
    "et": {
        "regression": ExtraTreesRegressor,
        "classification": ExtraTreesClassifier,
    },
    "adaboost": {
        "regression": AdaBoostRegressor,
        "classification": AdaBoostClassifier,
    },
    "bagging": {
        "regression": BaggingRegressor,
        "classification": BaggingClassifier,
    },
    "gradientboost": {
        "regression": GradientBoostingRegressor,
        "classification": GradientBoostingClassifier,
    },
    "stacking": {
        "classification": StackingClassifier,
        "regression": StackingRegressor,
    },
    # Gaussian Processes
    "gauss": {
        "regression": GaussianProcessRegressor,
        "classification": GaussianProcessClassifier,
    },
    # Linear models
    "logit": {
        "classification": LogisticRegression,
    },
    "logitcv": {
        "classification": LogisticRegressionCV,
    },
    "linreg": {
        "regression": LinearRegression,
    },
    "ridge": {
        "regression": Ridge,
        "classification": RidgeClassifier,
    },
    "ridgecv": {
        "regression": RidgeCV,
        "classification": RidgeClassifierCV,
    },
    "sgd": {
        "regression": SGDRegressor,
        "classification": SGDClassifier,
    },
    # Naive Bayes
    "nb_bernoulli": {
        "classification": BernoulliNB,
    },
    "nb_categorical": {
        "classification": CategoricalNB,
    },
    "nb_complement": {
        "classification": ComplementNB,
    },
    "nb_gaussian": {
        "classification": GaussianNB,
    },
    "nb_multinomial": {
        "classification": MultinomialNB,
    },
    # Dynamic Selection
    "ds": {
        "classification": DynamicSelection,
    },
    # Dummy
    "dummy": {
        "regression": DummyRegressor,
        "classification": DummyClassifier,
    },
}

_available_models_reset = deepcopy(_available_models)


def list_models() -> List[str]:
    """List all the available model names.

    Returns
    -------
    list of str
        A list will all the available model names.

    """
    out = list(_available_models.keys())
    return out


def get_model(name: str, problem_type: str, **kwargs: Any) -> ModelLike:
    """Get a model.

    Parameters
    ----------
    name : str
        The model name
    problem_type : str
        The type of problem. See :func:`.run_cross_validation`.
    **kwargs : dict
        Extra keyword arguments.

    Returns
    -------
    ModelLike
        The model object.

    """
    if name not in _available_models:
        raise_error(
            f"The specified model ({name}) is not available. "
            f"Valid options are: {list(_available_models.keys())}"
        )

    if problem_type not in _available_models[name]:
        raise_error(
            f"The specified model ({name})) is not suitable for"
            f"{problem_type}"
        )
    out = _available_models[name][problem_type](**kwargs)  # type: ignore
    return out


def register_model(
    model_name: str,
    classification_cls: Optional[Type[ModelLike]] = None,
    regression_cls: Optional[Type[ModelLike]] = None,
    overwrite: Optional[bool] = None,
):
    """Register a model to julearn.

    This function allows you to add a model or models for different problem
    types to julearn. Afterwards, it behaves like every other julearn model and
    can be referred to by name. E.g. you can use inside of
    `run_cross_validation` using `model=model_name`.

    Parameters
    ----------
    model_name : str
        Name by which model will be referenced by
    classification_cls : ModelLike
        The class which will be used for classification.
    regression_cls : ModelLike
        The class which will be used for regression.
    overwrite : bool, optional
        decides whether overwrite should be allowed

        Options are:

        * None : overwrite is possible, but warns the user (default).
        * True : overwrite is possible without any warning.
        * False : overwrite is not possible, error is raised instead.

    Raises
    ------
    ValueError
        If `model_name` is already registered and `overwrite` is False.

    Warns
    -----
    RuntimeWarning
        If `model_name` is already registered and `overwrite` is None.
    """
    problem_types = ["classification", "regression"]
    for cls, problem_type in zip(
        [classification_cls, regression_cls], problem_types
    ):
        if cls is not None:
            if (t_available := _available_models.get(model_name)) is not None:
                if t_available.get(problem_type):
                    if overwrite is None:
                        warn_with_log(
                            f"Model named {model_name} with"
                            " problem type {problem_type}"
                            " already exists. "
                            f"Therefore, {model_name} will be overwritten. "
                            "To remove this warning set overwrite=True. "
                            "If you won't to reset this use "
                            "`julearn.estimators.reset_model_register`."
                        )
                    elif overwrite is False:
                        raise_error(
                            f"Model named {model_name} with "
                            "problem type {problem_type}"
                            " already exists. "
                            f"Therefore, {model_name} will be overwritten. "
                            "overwrite is set to False, "
                            "therefore you cannot overwrite "
                            "existing models. Set overwrite=True"
                            " in case you want to "
                            "overwrite existing models"
                        )

                    logger.info(
                        f"registering model named {model_name} "
                        f"with problem_type {problem_type}"
                    )

                _available_models[model_name][problem_type] = cls
            else:
                logger.info(
                    f"registering model named {model_name} "
                    f"with problem_type {problem_type}"
                )
                _available_models[model_name] = {problem_type: cls}


def reset_model_register() -> None:
    """Reset the model register to the default state."""
    global _available_models
    _available_models = deepcopy(_available_models_reset)
