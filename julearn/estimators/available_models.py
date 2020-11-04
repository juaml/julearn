from sklearn.svm import SVC, SVR
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                              ExtraTreesClassifier, ExtraTreesRegressor)
from sklearn.dummy import DummyClassifier, DummyRegressor

from .. utils import raise_error

_available_models = {
    'svm': {
        'regression': SVR,
        'binary_classification': SVC,
        'multiclass_classification': SVC
    },
    'rf': {
        'regression': RandomForestRegressor,
        'binary_classification': RandomForestClassifier,
        'multiclass_classification': RandomForestClassifier
    },
    'et': {
        'regression': ExtraTreesRegressor,
        'binary_classification': ExtraTreesClassifier,
        'multiclass_classification': ExtraTreesClassifier
    },
    'dummy': {
        'regression': DummyRegressor,
        'binary_classification': DummyClassifier,
        'multiclass_classification': DummyClassifier,
    },
}


def list_models():
    """List all the available model names"""
    out = out = list(_available_models.keys)
    return out


def get_model(name, problem_type):
    """Get a model

    Parameters
    ----------
    name : str
        The model name
    problem_type : str
        The type of problem. See :func:`.run_cross_validation`.

    Returns
    -------
    out : scikit-learn compatible model
        The model object.

    """
    if name not in _available_models:
        raise_error(
            f'The specified model ({name}) is not available. '
            f'Valid options are: {list(_available_models.keys())}')

    if problem_type not in _available_models[name]:
        raise_error(
            f'The specified model ({name})) is not suitable for'
            f'{problem_type}')

    out = _available_models[name][problem_type]()
    return out
