from typing import List, Union, Optional

from sklearn.model_selection import BaseCrossValidator

import pandas as pd

from ..api import _compute_cvmdsum
from ..utils.logging import raise_error


_valid_funcs = [
    "predict",
    "predict_proba",
    "predict_log_proba",
    "decision_function",
]


def fold_predictions(
    scores: pd.DataFrame,
    cv: BaseCrossValidator,
    X: Union[str, List[str]],
    y : str,
    data: pd.DataFrame,
    func: str = "predict",
    groups : Optional[str] = None,
    pos_labels : Optional[Union[str, List[str]]] = None,
):
    if func not in _valid_funcs:
        raise_error(f"func must be one of {_valid_funcs}. Got {func}.")

    if "cv_mdsum" not in scores:
        raise_error(
            "The scores DataFrame must be the output of "
            "`run_cross_validation`. It is missing the `cv_mdsum` column."
        )

    cv_mdsums = scores["cv_mdsum"].unique()
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

    # Prepare data
    df_X, y, df_groups, X_types = prepare_input_data(
        X=X,
        y=y,
        df=data,
        pos_labels=pos_labels,
        groups=groups,
        X_types=X_types,
    )


    predictions = []
    for _, test in cv.split(df_X, y, groups=df_groups):
        predictions.append(getattr(model, func)(df_X.iloc[test]))