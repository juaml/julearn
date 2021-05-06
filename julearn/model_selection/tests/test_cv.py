# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
# License: AGPL
from julearn.model_selection.cv import (StratifiedGroupsKFold,
                                        RepeatedStratifiedGroupsKFold)
import numpy as np
from numpy.testing._private.utils import assert_array_equal
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from julearn.model_selection import StratifiedBootstrap


def test_stratified_bootstrap():
    """Test stratified bootstrap CV generator"""
    n_samples = 100
    X = np.random.rand(n_samples, 2)

    cases = [
        (3, .2),
        (2, .5),
        (4, .8),
    ]

    for n_classes, test_size in cases:
        y = np.random.randint(0, n_classes, n_samples)

        cv = StratifiedBootstrap(n_splits=10, test_size=test_size)

        for train, test in cv.split(X, y):
            y_train = y[train]
            y_test = y[test]
            for i in range(n_classes):
                n_y = (y == i).sum()
                n_y_train = (y_train == i).sum()
                n_y_test = (y_test == i).sum()
                print(n_y_test)
                print(n_y_train)
                assert abs(n_y_train - (n_y * (1 - test_size))) < 1
                assert abs(n_y_test - (n_y * test_size)) < 1


def test_stratified_groups_kfold():
    n_samples, n_features = 200, 20
    X = np.random.rand(n_samples, n_features)
    y = np.random.rand(n_samples)
    bins = np.digitize(y, bins=[0, 0.2, 0.4, 0.6, 0.8, 1])

    skcv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    jucv = StratifiedGroupsKFold(n_splits=3, shuffle=True, random_state=42)
    for (sk_train, sk_test), (ju_train, ju_test) in zip(
            skcv.split(X, bins), jucv.split(X, y, groups=bins)):
        assert_array_equal(sk_train, ju_train)
        assert_array_equal(sk_test, ju_test)

    skcv = RepeatedStratifiedKFold(n_repeats=4, n_splits=3, random_state=42)
    jucv = RepeatedStratifiedGroupsKFold(
        n_repeats=4, n_splits=3, random_state=42)
    for (sk_train, sk_test), (ju_train, ju_test) in zip(
            skcv.split(X, bins), jucv.split(X, y, groups=bins)):
        assert_array_equal(sk_train, ju_train)
        assert_array_equal(sk_test, ju_test)
