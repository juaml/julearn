"""Utility functions for model selection in julearn."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
# License: AGPL

from typing import TYPE_CHECKING

from sklearn.model_selection import check_cv as sk_check_cv

from .final_model_cv import _JulearnFinalModelCV


if TYPE_CHECKING:
    from ..utils.typing import CVLike


def check_cv(
    cv: "CVLike", classifier: bool = False, include_final_model: bool = False
) -> "CVLike":
    """Check the CV instance and return the proper CV for julearn.

    Parameters
    ----------
    cv : int, str or cross-validation generator | None
        Cross-validation splitting strategy to use for model evaluation.

        Options are:

        * None: defaults to 5-fold
        * int: the number of folds in a `(Stratified)KFold`
        * CV Splitter (see scikit-learn documentation on CV)
        * An iterable yielding (train, test) splits as arrays of indices.

    classifier : bool, default=False
        Whether the task is a classification task, in which case
        stratified KFold will be used.

    include_final_model : bool, default=False
        Whether to include the final model in the cross-validation. If true,
        one more fold will be added to the cross-validation, where the full
        dataset is used for training and testing

    Returns
    -------
    checked_cv : a cross-validator instance.
        The return value is a cross-validator which generates the train/test
        splits via the ``split`` method.

    """

    cv = sk_check_cv(cv, classifier=classifier)
    if include_final_model:
        cv = _JulearnFinalModelCV(cv)

    return cv
