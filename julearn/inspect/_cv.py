import warnings

from typing import List, Union, Optional

from sklearn.model_selection import BaseCrossValidator, check_cv

import pandas as pd

from ..utils import raise_error, _compute_cvmdsum, is_nonoverlapping_cv
from ._pipeline import PipelineInspector


_valid_funcs = [
    "predict",
    "predict_proba",
    "predict_log_proba",
    "decision_function",
]


class FoldsInspector:
    def __init__(
        self,
        scores: pd.DataFrame,
        cv: BaseCrossValidator,
        X: Union[str, List[str]],
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

        if "cv_mdsum" not in self._scores:
            raise_error(
                "The scores DataFrame must be the output of "
                "`run_cross_validation`. It is missing the `cv_mdsum` column."
            )

        cv_mdsums = self._scores["cv_mdsum"].unique()
        if cv_mdsums.size > 1:
            raise_error(
                "The scores CVs are not the same. Can't reproduce the CV folds."
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

    def predict_proba(self):
        return self._get_predictions("predict_proba")

    def predict_log_proba(self):
        return self._get_predictions("predict_log_proba")

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
            t_series = pd.Series(t_values, index=test)
            t_series.name = f"fold_{i_fold}"
            predictions.append(t_series)

        if is_nonoverlapping_cv(self._cv):
            n_repeats = self._scores["repeat"].unique().size
            n_folds = self._scores["fold"].unique().size
            folded_predictions = []
            for t_repeat in range(n_repeats):
                t_repeat_predictions = []
                for t_fold in range(n_folds):
                    t_repeat_predictions.append(
                        predictions[t_repeat * n_folds + t_fold])
                t_series = pd.concat(t_repeat_predictions, axis=0)
                t_series.name = f"repeat_{t_repeat}"
                folded_predictions.append(t_series)
            predictions = folded_predictions
        predictions = pd.concat(predictions, axis=1)
        return predictions.sort_index()

    def __getitem__(self, key):
        return FoldInspector(self, key)


class FoldInspector:
    def __init__(self, folds_inspector: FoldsInspector, i_fold: int):
        self._folds_inspector = folds_inspector
        self._i_fold = i_fold

    @property
    def model(self):
        return PipelineInspector(
            self._folds_inspector._scores["estimator"][self._i_fold]
        )
