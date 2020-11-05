# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL
import numpy as np
from numpy.testing import assert_array_equal
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier,
                              ExtraTreesClassifier)
from sklearn.dummy import DummyClassifier

from sklearn.base import clone
from sklearn.model_selection import cross_val_score

from julearn import run_cross_validation
from julearn.prepare import prepare_cv


def compare_models(clf1, clf2):
    if isinstance(clf1, SVC):
        idx1 = np.argsort(clf1.support_)
        v1 = clf1.support_vectors_[idx1]
        idx2 = np.argsort(clf2.support_)
        v2 = clf2.support_vectors_[idx2]
    elif isinstance(clf1, (RandomForestClassifier, ExtraTreesClassifier)):
        v1 = clf1.feature_importances_
        v2 = clf1.feature_importances_
    elif isinstance(clf1, DummyClassifier):
        v1 = None
        v2 = None
        assert clf1._strategy == clf2._strategy
        assert_array_equal(clf1.class_prior_, clf2.class_prior_)
        assert_array_equal(clf1.classes_, clf2.classes_)

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

    for scoring in scorers:
        np.random.seed(42)
        actual, actual_estimator = run_cross_validation(
            X=X, y=y, data=data, scoring=scoring, cv=cv,
            return_estimator=True, **api_params)

        np.random.seed(42)
        sk_cv = prepare_cv(cv)
        expected = cross_val_score(sklearn_model, sk_X, sk_y, cv=sk_cv,
                                   scoring=scoring)

        assert len(actual) == len(expected)
        assert all([a == b for a, b in zip(actual, expected)])

        # Compare the models
        clf1 = actual_estimator.dataframe_pipeline.steps[-1][1]
        clf2 = clone(sklearn_model).fit(sk_X, sk_y).steps[-1][1]
        compare_models(clf1, clf2)
