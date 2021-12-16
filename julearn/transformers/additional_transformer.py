# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL
import numpy as np
from joblib import Parallel, delayed
from scipy.stats import pearsonr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.fixes import _joblib_parallel_args
from sklearn.utils import check_X_y, check_array
from ..utils import warn


class CBPM(BaseEstimator, TransformerMixin):
    '''Transformer meaning all significantly to the target
    correlated features together.
    Significant negative and positive correlations are meaned separately.
    Non significant once are dropped. One can only chose to use negative
    or positive correlations. The others are then also dropped.
    In case that there are no significant correlations the mean target
    will be returned as the only feature.

    This transformer implements the procedure described in :
    Shen, X., Finn, E., Scheinost, D. et al. 2016
    https://doi.org/10.1038/nprot.2016.178

    Parameters
    ----------
    significance_threshold : float, default=0.05
        Threshold of p value.

    corr_method : callable, default=scipy.stats.pearsonr
        Callable which can be used to create tuple of arrays: (correlations,
        p values). Input has to be X, y.

    corr_values_used : str ,default='posneg'
        Which correlations should be used:
        Only positives use `'pos'`.
        Only negatives use `'neg'`.
        All use posneg.
        In case you use posneg and theere are only pos or neg this
        will be used instead. The actually used correlation_values can be
        found in the attribute: `used_corr_values_used_`

    weight_by_corr : bool, default=False
        If meaning the features should be weighted
        by each features correlation.

    n_jobs : int, default=None
        How many jobs should run in parallel to compute the correlations of
        each feature to the target.
        This parameter follows joblib and scikit-learn standards.

    verbose : int, default=0
        How verbose should the log of the parallel computing be.
        This parameter follows joblib and scikit-learn standards.

    Attributes
    ----------
    y_average_ : np.float64, by default None
        In case no significant correlation is present this value will
        safe the mean target to use it for the transformation.
        Else it will be None

    used_corr_values_used_ : str
        This will show you whether pos, neg or posneg was applied.
        See Parameter: corr_values_used

    X_y_correlations_ : tuple(np.array, np.array)
        Output of the corr_method. tuple(correlations, pvals).

    sign_mask_ : np.array of bools
        Array of bools showing which of the original features had a
        significant correlation.

    pos_correction_mask_ : np.array of bools
        Array of bools showing which of the original features had a
        positive correlation.

   '''

    def __init__(self, significance_threshold=0.05,
                 corr_method=pearsonr, corr_values_used='posneg',
                 weight_by_corr=False,
                 n_jobs=None, verbose=0):
        self.significance_threshold = significance_threshold
        self.corr_method = corr_method
        self.corr_values_used = corr_values_used
        self.weight_by_corr = weight_by_corr
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, X, y):

        X, y = check_X_y(X, y)
        self.X_y_correlations_ = np.array(Parallel(
            n_jobs=self.n_jobs, verbose=self.verbose,
            **_joblib_parallel_args())(
            delayed(self.corr_method)(X[:, X_idx], y)
            for X_idx in range(X.shape[1])
        ))

        self.create_masks(y)

        return self

    def transform(self, X):

        X = check_array(X)
        if self.y_average_ is not None:
            out = np.empty(X.shape[0])
            out.fill(self.y_average_)
            return out

        else:

            if self.used_corr_values_used_ == 'posneg':
                mask = self.sign_mask_

                X_meaned_pos = self.average(
                    X, mask=(self.sign_mask_ & self.pos_correction_mask_)
                )

                X_meaned_neg = self.average(
                    X, mask=(self.sign_mask_ & ~self.pos_correction_mask_)
                )

                X_meaned = np.concatenate(
                    [X_meaned_pos.reshape(-1, 1),
                     X_meaned_neg.reshape(-1, 1)],
                    axis=1)

            elif self.used_corr_values_used_ == 'pos':
                mask = self.sign_mask_ & self.pos_correction_mask_
                X_meaned = self.average(X, mask)

            elif self.used_corr_values_used_ == 'neg':
                mask = self.sign_mask_ & ~self.pos_correction_mask_
                X_meaned = self.average(X, mask)

            return X_meaned

    def create_masks(self, y):

        self.sign_mask_ = self.X_y_correlations_[
            :, 1] < self.significance_threshold
        self.pos_correction_mask_ = self.X_y_correlations_[:, 0] >= 0

        if self.corr_values_used == 'posneg':
            sig_feat_mask = self.sign_mask_
        elif self.corr_values_used == 'pos':
            sig_feat_mask = self.sign_mask_ & self.pos_correction_mask_
        elif self.corr_values_used == 'neg':
            sig_feat_mask = self.sign_mask_ & ~self.pos_correction_mask_
        if all(~sig_feat_mask):
            warn('No feature is significant. Therefore, the mean of'
                 ' target will be used for prediction instead.'
                 )
            self.y_average_ = y.mean()
            self.used_corr_values_used_ = None

        else:
            self.y_average_ = None

            if self.corr_values_used == 'posneg':
                if all(~(self.sign_mask_ & self.pos_correction_mask_)):
                    warn(
                        'No feature with significant positive correlations. '
                        'Only features with negative correlations will be '
                        'used. To get rid of this message, '
                        'set `corr_values_used = "neg".`'
                    )
                    self.used_corr_values_used_ = 'neg'

                elif all(~(self.sign_mask_
                           & ~self.pos_correction_mask_)):
                    warn(
                        'No feature with significant negative correlations. '
                        'Only features with positive correlations will be '
                        'used. To get rid of this message, '
                        'set `corr_values_used = "pos"`.'
                    )
                    self.used_corr_values_used_ = 'pos'
                else:
                    self.used_corr_values_used_ = self.corr_values_used
            else:
                self.used_corr_values_used_ = self.corr_values_used

    def average(self, X, mask):
        weights = (
            self.X_y_correlations_[:, 0][mask] if self.weight_by_corr
            else None
        )
        return np.average(X[:, mask], weights=weights, axis=1)
