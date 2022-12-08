"""Provide target confound removal."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL
from typing import Optional
from sklearn.base import clone
from sklearn.linear_model import LinearRegression

from .ju_target_transformer import JuTargetTransformer
from ...utils.typing import ModelLike


class TargetConfoundRemover(JuTargetTransformer):
    """Remove confounds from the target.

    Parameters
    ----------
    model_confound : ModelLike, optional
        Sklearn compatible model used to predict specified features
        independently using the confounds as features. The predictions of
        these models are then subtracted from each of the specified
        features, defaults to LinearRegression().
    confounds : str or list of str, optional
        The name of the 'confounds' type(s), i.e. which column type(s)
        represents the confounds. By default this is set to 'confounds'.
    threshold : float, optional
        All residual values after confound removal which fall under the
        threshold will be set to 0. None (default) means that no threshold
        will be applied.
    """

    def __init__(
        self,
        model_confound: Optional[ModelLike] = None,
        confounds: str = "confound",
        threshold: Optional[float] = None,
    ):
        if model_confound is None:
            model_confound = LinearRegression()
        self.model_confound = model_confound
        self.confounds = confounds
        self.threshold = threshold

    def fit(self, X, y=None):
        """Fit ConfoundRemover.

        Parameters
        ----------
        X : pd.DataFrame
            Training data for the confound remover.
        y : pd.Series, optional
            Training target values.

        Returns
        -------
        self : returns an instance of self.
        """
        self.confounds = self._ensure_column_types(self.confounds)

        self.model_confounds_ = clone(self.model_confound)
        self.detected_confounds_ = self.confounds.to_type_selector()(X)
        X_confounds = X.loc[:, self.detected_confounds_]
        self.model_confounds_.fit(X_confounds.values, y)
        return self

    def transform(self, X, y):
        """Remove confounds from the target.

        Parameters
        ----------
        X : pd.DataFrame
            Testing data for the confound remover.
        y : pd.Series, optional
            Target values.

        """
        X_confounds = X.loc[:, self.detected_confounds_]
        y_pred = self.model_confounds_.predict(X_confounds.values)
        residuals = y - y_pred
        if self.threshold is not None:
            residuals[abs(residuals) < self.threshold] = 0
        return residuals
