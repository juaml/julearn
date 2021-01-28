# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL
from sklearn.svm import SVC, SVR
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                              ExtraTreesClassifier, ExtraTreesRegressor,
                              AdaBoostClassifier, AdaBoostRegressor,
                              BaggingClassifier, BaggingRegressor,
                              GradientBoostingClassifier,
                              GradientBoostingRegressor)
from sklearn.gaussian_process import (GaussianProcessClassifier,
                                      GaussianProcessRegressor)
from sklearn.linear_model import (LogisticRegression, LogisticRegressionCV,
                                  LinearRegression, Ridge, RidgeClassifier,
                                  RidgeCV, RidgeClassifierCV,
                                  SGDRegressor, SGDClassifier)
from sklearn.naive_bayes import (BernoulliNB, CategoricalNB, ComplementNB,
                                 GaussianNB, MultinomialNB)
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
    'gauss': {
        'regression': GaussianProcessRegressor,
        'binary_classification': GaussianProcessClassifier,
        'multiclass_classification': GaussianProcessClassifier
    },
    'logit': {
        'binary_classification': LogisticRegression,
        'multiclass_classification': LogisticRegression,
    },
    'logitcv': {
        'binary_classification': LogisticRegressionCV,
        'multiclass_classification': LogisticRegressionCV,
    },
    'linreg': {
        'regression': LinearRegression,
    },
    'ridge': {
        'regression': Ridge,
        'binary_classification': RidgeClassifier,
        'multiclass_classification': RidgeClassifier,
    },
    'ridgecv': {
        'regression': RidgeCV,
        'binary_classification': RidgeClassifierCV,
        'multiclass_classification': RidgeClassifierCV,
    },
    'sgd': {
        'regression': SGDRegressor,
        'binary_classification': SGDClassifier,
        'multiclass_classification': SGDClassifier,
    },
    'adaboost': {
        'regression': AdaBoostRegressor,
        'binary_classification': AdaBoostClassifier,
        'multiclass_classification': AdaBoostClassifier,
    },
    'bagging': {
        'regression': BaggingRegressor,
        'binary_classification': BaggingClassifier,
        'multiclass_classification': BaggingClassifier,
    },
    'gradientboost': {
        'regression': GradientBoostingRegressor,
        'binary_classification': GradientBoostingClassifier,
        'multiclass_classification': GradientBoostingClassifier,
    },
    'nb_bernoulli': {
        'binary_classification': BernoulliNB,
        'multiclass_classification': BernoulliNB,
    },
    'nb_categorical': {
        'binary_classification': CategoricalNB,
        'multiclass_classification': CategoricalNB,
    },
    'nb_complement': {
        'binary_classification': ComplementNB,
        'multiclass_classification': ComplementNB,
    },
    'nb_gaussian': {
        'binary_classification': GaussianNB,
        'multiclass_classification': GaussianNB,
    },
    'nb_multinomial': {
        'binary_classification': MultinomialNB,
        'multiclass_classification': MultinomialNB,
    },
}


def list_models():
    """List all the available model names

    Returns
    -------
    out : list(str)
        A list will all the available model names.

    """
    out = list(_available_models.keys())
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
