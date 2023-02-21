# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
#          Kaustubh, Patil <k.patil@fz-juelich.de>
# License: AGPL

from typing import Callable, Optional

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import pearsonr
from sklearn.base import BaseEstimator, TransformerMixin
from ..utils.versions import _joblib_parallel_args
from ..utils import warn_with_log


class CBPM(BaseEstimator, TransformerMixin):
    """Connectome-Based Predictive Modeling transformer.

    This transformer means together all features significantly
    correlated to the target.

    Significant negative and positive correlations are meaned separately.
    Non-significant ones are dropped.

    User can choose to use negative, positive or both correlations.

    In case that there are no significant correlations the mean of the
    target will be returned as the only feature.

    This transformer implements the procedure described in :
    Shen, X., Finn, E., Scheinost, D. et al. 2016
    https://doi.org/10.1038/nprot.2016.178

    Parameters
    ----------
    significance_threshold : float, default=0.05
        Threshold of p value.
    corr_method : callable, default=scipy.stats.pearsonr
        Callable which can be used to create tuple of arrays: (correlations,
        p values). Input has to be two arrays of the same length.
    corr_sign : str , default='posneg'
        Which correlations should be used:
        Options are:

         * `pos`: use positive correlations only
         * `neg`: use negative correlations only
         * `posneg`: use all correlations

        In case you use `posneg` and there are only `pos` or `neg` this
        will be used instead. The actually used correlation_values can be
        found in the attribute: `used_corr_sign_`
    weight_by_corr : bool, default=False
        If True, the mean of each feature will be weighted by each its
        correlation value.
    n_jobs : int, default=None
        How many jobs should run in parallel to compute the correlations of
        each feature to the target. This parameter follows joblib and
        scikit-learn standards.
    verbose : int, default=0
        How verbose should the log of the parallel computing be. This parameter
        follows joblib and scikit-learn standards.

    Attributes
    ----------
    y_mean_ : np.array
        Contans the mean of the target, to be used for the transformation in
        case that there are no significant correlations.
    used_corr_sign_ : str
        This will show you whether pos, neg or posneg was applied.
    X_y_correlations_ : tuple(np.array, np.array)
        Output of the `corr_method` applied to the target and the features.
    significant_mask_ : np.array of bools
        Array of bools showing which of the original features had a
        significant correlation.
    pos_mask_ : np.array of bools
        Array of bools showing which of the original features had a
        positive correlation.
    pos_significant_mask_ : np.array of bools
        Array of bools showing which of the original features had a
        significant positive correlation.
    neg_significant_mask_ : np.array of bools
        Array of bools showing which of the original features had a
        significant negative correlation.
    used_significant_mask_ : np.array of bools
        Array of bools showing which of the original features will be used
        by this transformer.
    """

    def __init__(
        self,
        significance_threshold: float = 0.05,
        corr_method: Callable = pearsonr,
        corr_sign: str = "posneg",
        weight_by_corr: bool = False,
        n_jobs: Optional[int] = None,
        verbose: int = 0,
    ):
        self.significance_threshold = significance_threshold
        self.corr_method = corr_method
        if corr_sign not in ["pos", "neg", "posneg"]:
            raise ValueError(
                "corr_sign must be one of pos, neg or posneg, "
                f"but is {corr_sign}"
            )
        self.corr_sign = corr_sign
        self.weight_by_corr = weight_by_corr
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CBPM":
        """Fit the transformer.

        Compute the correlations of each feature to the target, threhsold and
        create the respective masks.

        Parameters
        ----------
        X : np.array
            Input features.
        y : np.array
            Target.

        Returns
        -------
        self : CBPM
            The fitted transformer.
        """
        X, y = self._validate_data(X, y)  # type: ignore

        # compute correlations using joblib
        self.X_y_correlations_ = np.array(
            Parallel(
                n_jobs=self.n_jobs,
                verbose=self.verbose,
                **_joblib_parallel_args(),
            )(
                delayed(self.corr_method)(X[:, X_idx], y)
                for X_idx in range(X.shape[1])
            )
        )

        # Save the y mean to be used in transform in case no significant
        # correlation is present
        self.y_mean_ = y.mean()

        # Create the masks
        self._create_masks()

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform the data.

        Replace each of the features that had a significant correlation on
        the training data with the mean of the features (mean is computed
        per sample).

        Parameters
        ----------
        X : np.array
            Input features.

        Returns
        -------
        np.array
            The transformed features.
        """

        X = self._validate_data(X)  # type: ignore

        if not any(self.used_significant_mask_):
            out = np.ones(X.shape[0]) * self.y_mean_
            return out

        if self.used_corr_sign_ == "pos":
            X_meaned = self._average(X, self.pos_significant_mask_)
        elif self.used_corr_sign_ == "neg":
            X_meaned = self._average(X, self.neg_significant_mask_)
        else:
            X_meaned_pos = self._average(X, mask=self.pos_significant_mask_)
            X_meaned_neg = self._average(X, mask=self.neg_significant_mask_)

            X_meaned = np.concatenate(
                [X_meaned_pos.reshape(-1, 1), X_meaned_neg.reshape(-1, 1)],
                axis=1,
            )

        return X_meaned

    def _create_masks(self) -> None:
        """Create the masks for the significant correlations."""

        # Find the correlations whose p-value is below the threshold
        self.significant_mask_ = (
            self.X_y_correlations_[:, 1] < self.significance_threshold
        )

        # Mask separately the positive and negative correlations
        self.pos_mask_ = self.X_y_correlations_[:, 0] > 0
        self.neg_mask_ = self.X_y_correlations_[:, 0] < 0

        # Mask the significant correlations separately by sign
        self.pos_significant_mask_ = np.logical_and(
            self.significant_mask_, self.pos_mask_
        )
        self.neg_significant_mask_ = np.logical_and(
            self.significant_mask_, self.neg_mask_
        )

        have_pos_feat = any(self.pos_significant_mask_)
        have_neg_feat = any(self.neg_significant_mask_)

        if self.corr_sign == "pos":
            self.used_corr_sign_ = "pos"
            self.used_significant_mask_ = self.pos_significant_mask_
            if not have_pos_feat:
                warn_with_log(
                    "No feature with significant positive correlations was "
                    "present. Therefore, the mean of the target will be used "
                    "for prediction instead."
                )
        elif self.corr_sign == "neg":
            self.used_corr_sign_ = "neg"
            self.used_significant_mask_ = self.neg_significant_mask_
            if not have_neg_feat:
                warn_with_log(
                    "No feature with significant negative correlations was "
                    "present. Therefore, the mean of the target will be used "
                    "for prediction instead."
                )
        else:  # "posneg"
            if not have_pos_feat and not have_neg_feat:
                self.used_corr_sign_ = "posneg"
                self.used_significant_mask_ = np.zeros(
                    self.significant_mask_.shape, dtype=bool
                )
                warn_with_log(
                    "No feature with significant negative or positive "
                    "correlations was present. Therefore, the mean of the "
                    "target will be used for prediction instead."
                )
            elif not have_pos_feat:
                self.used_corr_sign_ = "neg"
                self.used_significant_mask_ = self.neg_significant_mask_
                warn_with_log(
                    "No feature with significant positive correlations was "
                    "present. Only features with negative correlations will "
                    "be used. To get rid of this message, set "
                    "`corr_sign = 'neg'`."
                )
            elif not have_neg_feat:
                self.used_corr_sign_ = "pos"
                self.used_significant_mask_ = self.pos_significant_mask_
                warn_with_log(
                    "No feature with significant negative correlations was "
                    "present. Only features with positive correlations will "
                    "be used. To get rid of this message, set "
                    "`corr_sign = 'pos'`."
                )
            else:
                self.used_corr_sign_ = "posneg"
                self.used_significant_mask_ = self.significant_mask_

    def _average(self, X: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Compute the per-sample average of features.

        Parameters
        ----------
        X : np.array
            Input features.
        mask : np.array
            Mask of the features to be averaged.

        Returns
        -------
        np.array
            The per-sample average of the features.
        """
        weights = (
            self.X_y_correlations_[:, 0][mask] if self.weight_by_corr else None
        )
        return np.average(X[:, mask], weights=weights, axis=1)
