# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL
import numpy as np
from sklearn.base import clone
from sklearn.svm import SVC, SVR
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                              ExtraTreesClassifier, ExtraTreesRegressor,
                              AdaBoostClassifier, AdaBoostRegressor,
                              BaggingClassifier, BaggingRegressor,
                              GradientBoostingClassifier,
                              GradientBoostingRegressor)
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.gaussian_process import (GaussianProcessClassifier,
                                      GaussianProcessRegressor)
from sklearn.linear_model import (LogisticRegression, LogisticRegressionCV,
                                  LinearRegression, Ridge, RidgeClassifier,
                                  RidgeCV, RidgeClassifierCV,
                                  SGDRegressor, SGDClassifier)
from sklearn.naive_bayes import (BernoulliNB, CategoricalNB, ComplementNB,
                                 GaussianNB, MultinomialNB)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from seaborn import load_dataset

import pytest

from julearn.utils.testing import do_scoring_test
from julearn.estimators import list_models, get_model


_clf_estimators = {
    'svm': SVC,
    'rf': RandomForestClassifier,
    'et': ExtraTreesClassifier,
    'dummy': DummyClassifier,
    'gauss': GaussianProcessClassifier,
    'logit': LogisticRegression,
    'logitcv': LogisticRegressionCV,
    'ridge': RidgeClassifier,
    'ridgecv': RidgeClassifierCV,
    'sgd': SGDClassifier,
    'adaboost': AdaBoostClassifier,
    'bagging': BaggingClassifier,
    'gradientboost': GradientBoostingClassifier,
}

_clf_params = {
    'rf': {'n_estimators': 10},
    'et': {'n_estimators': 10},
    'dummy': {'strategy': 'prior'},
    'sgd': {'random_state': 2},
}

_reg_estimators = {
    'svm': SVR,
    'rf': RandomForestRegressor,
    'et': ExtraTreesRegressor,
    'dummy': DummyRegressor,
    'gauss': GaussianProcessRegressor,
    'linreg': LinearRegression,
    'ridge': Ridge,
    'ridgecv': RidgeCV,
    'sgd': SGDRegressor,
    'adaboost': AdaBoostRegressor,
    'bagging': BaggingRegressor,
    'gradientboost': GradientBoostingRegressor
}

_reg_params = {
    'rf': {'n_estimators': 10},
    'et': {'n_estimators': 10},
    'dummy': {'strategy': 'mean'},
    'sgd': {'random_state': 2},
    'adaboost': {'random_state': 2},
    'bagging': {'random_state': 2}
}

_nb_estimators = {
    'nb_bernoulli': BernoulliNB,
    'nb_categorical': CategoricalNB,
    'nb_complement': ComplementNB,
    'nb_gaussian': GaussianNB,
    'nb_multinomial': MultinomialNB,
}

_nb_params = {
}


def test_naive_bayes_estimators():
    """Test all naive bayes estimators"""
    df_iris = load_dataset('iris')

    # keep only two species
    df_binary = df_iris[df_iris['species'].isin(['setosa', 'virginica'])]
    X = ['sepal_length', 'sepal_width', 'petal_length']
    y = 'species'

    for t_mname, t_model_class in _nb_estimators.items():
        m_params = _nb_params.get(t_mname, {})
        model_params = None
        if len(m_params) > 0:
            model_params = {
                f'{t_mname}__{t_param}': t_value
                for t_param, t_value in m_params.items()
            }
            t_model = t_model_class(**m_params)
        else:
            t_model = t_model_class()
        t_df_binary = df_binary.copy(deep=True)
        t_df = df_iris.copy(deep=True)
        if t_mname in ['nb_categorical']:
            t_df_binary[X] = t_df_binary[X] > t_df_binary[X].mean()
            t_df[X] = t_df[X] > t_df[X].mean()
        scorers = ['accuracy']
        api_params = {'model': t_mname, 'model_params': model_params,
                      'preprocess_X': None}
        clf = make_pipeline(clone(t_model))
        do_scoring_test(X, y, data=t_df_binary, api_params=api_params,
                        sklearn_model=clf, scorers=scorers)
        api_params = {'model': t_mname, 'model_params': model_params,
                      'preprocess_X': None,
                      'problem_type': 'multiclass_classification'}
        clf = make_pipeline(clone(t_model))
        do_scoring_test(X, y, data=t_df, api_params=api_params,
                        sklearn_model=clf, scorers=scorers)
        if t_mname not in ['nb_bernoulli']:
            # now let's try target-dependent scores
            scorers = ['recall', 'precision', 'f1']
            sk_y = (t_df_binary[y].values == 'setosa').astype(np.int64)
            api_params = {'model': t_mname, 'pos_labels': 'setosa',
                          'model_params': model_params,
                          'preprocess_X': None}
            clf = make_pipeline(clone(t_model))
            do_scoring_test(X, y, data=t_df_binary, api_params=api_params,
                            sklearn_model=clf,
                            scorers=scorers, sk_y=sk_y)


def test_binary_estimators():
    """Test all binary estimators"""
    df_iris = load_dataset('iris')

    # keep only two species
    df_iris = df_iris[df_iris['species'].isin(['setosa', 'virginica'])]
    X = ['sepal_length', 'sepal_width', 'petal_length']
    y = 'species'

    for t_mname, t_model_class in _clf_estimators.items():
        m_params = _clf_params.get(t_mname, {})
        model_params = None
        if len(m_params) > 0:
            model_params = {
                f'{t_mname}__{t_param}': t_value
                for t_param, t_value in m_params.items()
            }
            t_model = t_model_class(**m_params)
        else:
            t_model = t_model_class()
        scorers = ['accuracy']
        api_params = {'model': t_mname, 'model_params': model_params}
        clf = make_pipeline(StandardScaler(), clone(t_model))
        do_scoring_test(X, y, data=df_iris, api_params=api_params,
                        sklearn_model=clf, scorers=scorers)
        if t_mname != 'dummy':
            # now let's try target-dependent scores
            scorers = ['recall', 'precision', 'f1']
            sk_y = (df_iris[y].values == 'setosa').astype(np.int64)
            api_params = {'model': t_mname, 'pos_labels': 'setosa',
                          'model_params': model_params}
            clf = make_pipeline(StandardScaler(), clone(t_model))
            do_scoring_test(X, y, data=df_iris, api_params=api_params,
                            sklearn_model=clf,
                            scorers=scorers, sk_y=sk_y)


def test_multiclass_estimators():
    """Test all multiclass estimators"""
    df_iris = load_dataset('iris')

    X = ['sepal_length', 'sepal_width', 'petal_length']
    y = 'species'

    for t_mname, t_model_class in _clf_estimators.items():
        m_params = _clf_params.get(t_mname, {})
        model_params = None
        if len(m_params) > 0:
            model_params = {
                f'{t_mname}__{t_param}': t_value
                for t_param, t_value in m_params.items()
            }
            t_model = t_model_class(**m_params)
        else:
            t_model = t_model_class()
        scorers = ['accuracy']
        api_params = {'model': t_mname, 'model_params': model_params,
                      'problem_type': 'multiclass_classification'}
        clf = make_pipeline(StandardScaler(), clone(t_model))
        do_scoring_test(X, y, data=df_iris, api_params=api_params,
                        sklearn_model=clf, scorers=scorers)


def test_regression_estimators():
    """Test all regression estimators"""
    df_iris = load_dataset('iris')

    X = ['sepal_length', 'sepal_width', 'petal_length']
    y = 'petal_width'

    for t_mname, t_model_class in _reg_estimators.items():
        m_params = _reg_params.get(t_mname, {})
        model_params = None
        if len(m_params) > 0:
            model_params = {
                f'{t_mname}__{t_param}': t_value
                for t_param, t_value in m_params.items()
            }
            t_model = t_model_class(**m_params)
        else:
            t_model = t_model_class()
        scorers = ['neg_root_mean_squared_error', 'r2']
        api_params = {'model': t_mname, 'model_params': model_params,
                      'problem_type': 'regression'}
        clf = make_pipeline(StandardScaler(), clone(t_model))
        do_scoring_test(X, y, data=df_iris, api_params=api_params,
                        sklearn_model=clf, scorers=scorers)


def test_list_get_models():
    """Test list and getting models"""
    expected = set(
        list(_clf_estimators.keys()) + list(_nb_estimators.keys()) +
        list(_reg_estimators.keys()))
    expected.add('ds')
    actual = list_models()
    diff = set(actual) ^ set(expected)
    assert not diff

    expected = _clf_estimators['dummy']
    actual = get_model('dummy', 'binary_classification')

    assert isinstance(actual, expected)

    expected = _clf_estimators['dummy']
    actual = get_model('dummy', 'multiclass_classification')

    assert isinstance(actual, expected)

    expected = _reg_estimators['dummy']
    actual = get_model('dummy', 'regression')

    assert isinstance(actual, expected)

    with pytest.raises(ValueError, match="is not suitable for"):
        get_model('linreg', 'binary_classification')

    with pytest.raises(ValueError, match="is not available"):
        get_model('wrong', 'binary_classification')
