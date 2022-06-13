# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL
import warnings
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
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
from sklearn.linear_model import (LogisticRegression,
                                  LinearRegression, Ridge, RidgeClassifier,
                                  RidgeCV, RidgeClassifierCV,
                                  SGDRegressor, SGDClassifier)
from sklearn.naive_bayes import (BernoulliNB, CategoricalNB, ComplementNB,
                                 GaussianNB, MultinomialNB)
from sklearn.base import clone, TransformerMixin, BaseEstimator
from sklearn.model_selection import cross_validate

from julearn import run_cross_validation
from julearn.prepare import prepare_cv


def compare_models(clf1, clf2):  # pragma: no cover
    if isinstance(clf1, (SVC, SVR)):
        idx1 = np.argsort(clf1.support_)
        v1 = clf1.support_vectors_[idx1]
        idx2 = np.argsort(clf2.support_)
        v2 = clf2.support_vectors_[idx2]
    elif isinstance(clf1, (RandomForestClassifier, RandomForestRegressor,
                           ExtraTreesClassifier, ExtraTreesRegressor,
                           GradientBoostingClassifier,
                           GradientBoostingRegressor)):
        v1 = clf1.feature_importances_
        v2 = clf1.feature_importances_
    elif isinstance(clf1, (DummyClassifier, DummyRegressor)):
        v1 = None
        v2 = None
        if hasattr(clf1, '_strategy'):
            assert clf1._strategy == clf2._strategy
        if hasattr(clf1, 'strategy'):
            assert clf1.strategy == clf2.strategy
        if hasattr(clf1, 'class_prior_'):
            assert_array_equal(clf1.class_prior_, clf2.class_prior_)
        if hasattr(clf1, 'constant_'):
            assert clf1.constant_ == clf2.constant_
        if hasattr(clf1, 'classes_'):
            assert_array_equal(clf1.classes_, clf2.classes_)
    elif isinstance(clf1, GaussianProcessClassifier):
        if hasattr(clf1.base_estimator_, 'estimators_'):
            # Multiclass
            est1 = clf1.base_estimator_.estimators_
            v1 = np.array([x.pi_ for x in est1])
            est2 = clf2.base_estimator_.estimators_
            v2 = np.array([x.pi_ for x in est2])
        else:
            v1 = clf1.base_estimator_.pi_
            v2 = clf2.base_estimator_.pi_
    elif isinstance(clf1, GaussianProcessRegressor):
        v1 = np.c_[clf1.L_, clf1.alpha_]
        v2 = np.c_[clf2.L_, clf2.alpha_]
    elif isinstance(clf1, (LogisticRegression, RidgeClassifier,
                           RidgeClassifierCV, SGDClassifier, SGDRegressor,
                           LinearRegression, Ridge, RidgeCV,
                           BernoulliNB, ComplementNB, MultinomialNB)):
        v1 = _get_coef_over_versions(clf1)
        v2 = _get_coef_over_versions(clf1)
    elif isinstance(clf1, CategoricalNB):
        v1 = None
        v2 = None
        for c1, c2 in zip(_get_coef_over_versions(clf1),
                          _get_coef_over_versions(clf2)):
            assert_array_equal(c1, c2)
    elif isinstance(clf1, GaussianNB):
        v1 = clf1.sigma_
        v2 = clf2.sigma_
    elif isinstance(clf1, (AdaBoostClassifier, AdaBoostRegressor,
                           BaggingClassifier, BaggingRegressor)):
        est1 = clf1.estimators_
        v1 = np.array([x.feature_importances_ for x in est1])
        est2 = clf2.estimators_
        v2 = np.array([x.feature_importances_ for x in est2])
    else:
        raise NotImplementedError(
            f'Model comparison for {clf1} not yet implemented.')
    assert_array_equal(v1, v2)


def do_scoring_test(X, y, data, api_params, sklearn_model, scorers, cv=None,
                    sk_y=None):

    if cv is None:
        cv = 'repeat:1_nfolds:2'
    sk_X = data[X].values
    if sk_y is None:
        sk_y = data[y].values

    np.random.seed(42)
    params_dict = {k: v for k, v in api_params.items()}
    if 'preprocess_X' not in params_dict:
        params_dict['preprocess_X'] = 'zscore'
    actual, actual_estimator = run_cross_validation(
        X=X, y=y, data=data, scoring=scorers, cv=cv,
        return_estimator='final', **params_dict)
    np.random.seed(42)
    sk_cv = prepare_cv(cv)
    expected = cross_validate(sklearn_model, sk_X, sk_y, cv=sk_cv,
                              scoring=scorers)

    for scoring in scorers:
        s_key = f'test_{scoring}'
        assert len(actual.columns) == len(expected) + 2
        assert len(actual[s_key]) == len(expected[s_key])
        assert_array_almost_equal(actual[s_key], expected[s_key], decimal=5)

        # Compare the models
        clf1 = actual_estimator.dataframe_pipeline.steps[-1][1]
        clf2 = clone(sklearn_model).fit(sk_X, sk_y).steps[-1][1]
        compare_models(clf1, clf2)


class PassThroughTransformer(TransformerMixin, BaseEstimator):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X


class TargetPassThroughTransformer(PassThroughTransformer):

    def __init__(self):
        """A target transformer doing nothing.
        It only returns the target as it is.

        """
        super().__init__()

    def transform(self, X=None, y=None):
        return y

    def fit_transform(self, X=None, y=None):
        self.fit(X, y)
        return self.transform(X, y)


def _get_coef_over_versions(clf):

    if isinstance(clf, (BernoulliNB, ComplementNB,
                        MultinomialNB, CategoricalNB)):

        with warnings.catch_warnings():
            warnings.filterwarnings('error', category=FutureWarning)
            warnings.filterwarnings('error', category=DeprecationWarning)
            return clf.feature_log_prob_
    else:
        return clf.coef_
