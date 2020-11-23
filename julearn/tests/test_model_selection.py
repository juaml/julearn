# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
# License: AGPL
import numpy as np
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
