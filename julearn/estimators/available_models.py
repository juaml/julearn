from sklearn.svm import SVC, SVR
from sklearn.ensemble import (RandomForestClassifier,  RandomForestRegressor,
                              ExtraTreesClassifier, ExtraTreesRegressor)
from sklearn.dummy import DummyClassifier, DummyRegressor

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
    out = out = list(_available_models.keys)
    return out


def get_model(name, problem_type):
    if name not in _available_models:
        raise ValueError(
            f'The specified model ({name}) is not available. '
            f'Valid options are: {list(_available_models.keys())}')

    if problem_type not in _available_models[name]:
        raise ValueError(
            f'The specified model ({name})) is not suitable for'
            f'{problem_type}')

    out = _available_models[name][problem_type]()
    return out
