"""Compute md5sum of a cross-validation object."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL


import numpy as np
import hashlib
import inspect
import json

from sklearn.model_selection import (
    GroupKFold,
    KFold,
    LeaveOneGroupOut,
    LeaveOneOut,
    RepeatedKFold,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    StratifiedGroupKFold,
)
from sklearn.model_selection._split import PredefinedSplit, _CVIterableWrapper
from ..model_selection import (
    RepeatedContinuousStratifiedGroupKFold,
    ContinuousStratifiedGroupKFold,
)


def _recurse_to_list(a):
    """Recursively convert a to a list."""
    if isinstance(a, (list, tuple)):
        return [_recurse_to_list(i) for i in a]
    elif isinstance(a, np.ndarray):
        return a.tolist()
    else:
        return a


def _compute_cvmdsum(cv):
    """Compute the sum of the CV generator."""
    params = {k: v for k, v in vars(cv).items()}
    params["class"] = cv.__class__.__name__

    out = None

    # Check for special cases that might not be comparable
    if "random_state" in params:
        if params["random_state"] is None:
            if params.get("shuffle", True) is True:
                # If it's shuffled and the random seed is None
                out = "non-reproducible"

    if isinstance(cv, _CVIterableWrapper):
        splits = params.pop("cv")
        params["cv"] = _recurse_to_list(splits)
    if isinstance(cv, PredefinedSplit):
        params["test_fold"] = params["test_fold"].tolist()
        params["unique_folds"] = params["unique_folds"].tolist()

    if "cv" in params:
        if inspect.isclass(params["cv"]):
            params["cv"] = params["cv"].__class__.__name__

    if out is None:
        out = hashlib.md5(
            json.dumps(params, sort_keys=True).encode("utf-8")
        ).hexdigest()

    return out


def is_nonoverlapping_cv(cv) -> bool:
    _valid_instances = (
        KFold,
        GroupKFold,
        RepeatedKFold,
        RepeatedStratifiedKFold,
        StratifiedKFold,
        LeaveOneOut,
        LeaveOneGroupOut,
        StratifiedGroupKFold,
        ContinuousStratifiedGroupKFold,
        RepeatedContinuousStratifiedGroupKFold,
    )

    return isinstance(cv, _valid_instances)
