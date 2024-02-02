"""Provide base classes and functions to inspect folds of cross-validation."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL

from typing import List, Optional, Union

import pandas as pd
from sklearn.model_selection import BaseCrossValidator, check_cv
from sklearn.utils.metaestimators import available_if

from ..utils import _compute_cvmdsum, is_nonoverlapping_cv, raise_error
from ._pipeline import PipelineInspector


_valid_funcs = [
    "predict",
    "predict_proba",
    "predict_log_proba",
    "decision_function",
]


def _wrapped_model_has(attr):
    """Create a function to check if self.model_ has a given attribute.

    This function is usable by
    :func:`sklearn.utils.metaestimators.available_if`

    Parameters
    ----------
    attr : str
        The attribute to check for.

    Returns
    -------
    check : function
        The check function.

    """

    def check(self):
        """Check if self.model_ has a given attribute.

        Returns
        -------
        bool
            True if first estimator in scores has the attribute,
            False otherwise.

        """
        model_ = self._scores["estimator"].iloc[0]
        return hasattr(model_, attr)

    return check


class FoldsInspector:
    def __init__(
        self,
        scores: pd.DataFrame,
        cv: BaseCrossValidator,
        X: Union[str, List[str]],  # noqa: N803
        y: str,
        func: str = "predict",
        groups: Optional[str] = None,
    ):
        self._scores = scores
        self._cv = cv
        self._X = X
        self._y = y
        self._func = func
        self._groups = groups

        self._current_fold = 0
        if "cv_mdsum" not in list(self._scores.columns):
            raise_error(
                "The scores DataFrame must be the output of "
                "`run_cross_validation`. It is missing the `cv_mdsum` column."
            )

        cv_mdsums = self._scores["cv_mdsum"].unique()
        if cv_mdsums.size > 1:
            raise_error(
                "The scores CVs are not the same."
                "Can't reproduce the CV folds."
            )
        if cv_mdsums[0] == "non-reproducible":
            raise_error(
                "The CV is non-reproducible. Can't reproduce the CV folds."
            )

        cv = check_cv(cv)

        t_cv_mdsum = _compute_cvmdsum(cv)
        if t_cv_mdsum != cv_mdsums[0]:
            raise_error(
                "The CVs are not the same. Can't reproduce the CV folds."
            )

    def predict(self):
        return self._get_predictions("predict")

    @available_if(_wrapped_model_has("predict_proba"))
    def predict_proba(self):
        return self._get_predictions("predict_proba")

    @available_if(_wrapped_model_has("predict_log_proba"))
    def predict_log_proba(self):
        return self._get_predictions("predict_log_proba")

    @available_if(_wrapped_model_has("decision_function"))
    def decision_function(self):
        return self._get_predictions("decision_function")

    def _get_predictions(self, func):
        if func not in _valid_funcs:
            raise_error(f"Invalid func: {func}")

        predictions = []
        for i_fold, (_, test) in enumerate(
            self._cv.split(self._X, self._y, groups=self._groups)
        ):
            t_model = self._scores["estimator"][i_fold]
            t_values = getattr(t_model, func)(self._X.iloc[test])
            if t_values.ndim == 1:
                t_values = t_values[:, None]
            column_names = [f"p{i}" for i in range(t_values.shape[1])]
            t_predictions_df = pd.DataFrame(
                t_values, index=test, columns=column_names
            )
            predictions.append(t_predictions_df)

        if is_nonoverlapping_cv(self._cv):
            n_repeats = self._scores["repeat"].unique().size
            n_folds = self._scores["fold"].unique().size
            folded_predictions = []
            for t_repeat in range(n_repeats):
                t_repeat_predictions = []
                for t_fold in range(n_folds):
                    t_repeat_predictions.append(
                        predictions[t_repeat * n_folds + t_fold]
                    )
                t_df = pd.concat(t_repeat_predictions, axis=0)
                prefix = f"repeat{t_repeat}"
                t_df.columns = [f"{prefix}_{c}" for c in t_df.columns]
                folded_predictions.append(t_df)
            predictions = folded_predictions
        else:
            for i_fold, t_df in enumerate(predictions):
                t_df.columns = [f"fold{i_fold}_{x}" for x in t_df.columns]
        predictions = pd.concat(predictions, axis=1)
        predictions = predictions.sort_index()
        predictions["target"] = self._y.values
        return predictions

    def __getitem__(self, key):
        return _FoldInspector(self, key)

    def __iter__(self):
        return self

    def __next__(self):
        if self._current_fold == self.__len__():
            raise StopIteration
        this_fold = self[self._current_fold]
        self._current_fold += 1
        return this_fold

    def __len__(self):
        return len(self._scores)


class _FoldInspector:
    def __init__(self, folds_inspector: FoldsInspector, i_fold: int):
        self._folds_inspector = folds_inspector
        self._i_fold = i_fold

    @property
    def model(self):
        return PipelineInspector(
            self._folds_inspector._scores["estimator"][self._i_fold]
        )
