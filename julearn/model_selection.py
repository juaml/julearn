# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL
import numpy as np

from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection._split import (BaseShuffleSplit,
                                            _validate_shuffle_split)


def wrap_search(searcher, *args, **kwargs):

    class wrap_searcher(searcher):

        def __init__(self):
            super().__init__(*args, **kwargs)

        def transform_target(self, X, y):
            check_is_fitted(self)
            return self.best_estimator_.transform_target(X, y)

    return wrap_searcher()


class StratifiedBootstrap(BaseShuffleSplit):
    """Stratified Bootstrap cross-validation iterator

    Provides train/test indices using resampling with replacement, respecting
    the distribution of samples for each class.

    Parameters
    ----------
    n_splits : int, default=5
        Number of re-shuffling & splitting iterations.
    test_size : float, int, default=0.2
        If float, should be between 0.0 and 1.0 and represent the proportion
        of groups to include in the test split (rounded up). If int,
        represents the absolute number of test groups. If None, the value is
        set to the complement of the train size.
        The default will change in version 0.21. It will remain 0.2 only
        if ``train_size`` is unspecified, otherwise it will complement
        the specified ``train_size``.
    train_size : float or int, default=None
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the groups to include in the train split. If
        int, represents the absolute number of train groups. If None,
        the value is automatically set to the complement of the test size.
    random_state : int or RandomState instance, default=None
        Controls the randomness of the training and testing indices produced.
        Pass an int for reproducible output across multiple function calls.
    """
    def __init__(self, n_splits=5, *, test_size=0.5, train_size=None,
                 random_state=None):
        super().__init__(
            n_splits=n_splits,
            test_size=test_size,
            train_size=train_size,
            random_state=random_state)

    def _iter_indices(self, X, y, groups=None):
        y_labels = np.unique(y)
        y_inds = [np.where(y == t_y)[0] for t_y in y_labels]
        n_samples = [
            _validate_shuffle_split(
                len(t_inds), self.test_size, self.train_size,
                default_test_size=self._default_test_size)
            for t_inds in y_inds]
        for _ in range(self.n_splits):
            train = []
            test = []
            for t_inds, (n_train, _) in zip(y_inds, n_samples):
                bs_inds = np.random.choice(t_inds, len(t_inds), replace=True)
                train.extend(bs_inds[:n_train])
                test.extend(bs_inds[n_train:])

            yield train, test

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
            Note that providing ``y`` is sufficient to generate the splits and
            hence ``np.zeros(n_samples)`` may be used as a placeholder for
            ``X`` instead of actual training data.
        y : array-like of shape (n_samples,) or (n_samples, n_labels)
            The target variable for supervised learning problems.
            Stratification is done based on the y labels.
        groups : object
            Always ignored, exists for compatibility.

        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.

        Notes
        -----
        Randomized CV splitters may return different results for each call of
        split. You can make the results identical by setting `random_state`
        to an integer.
        """
        return super().split(X, y, groups)
